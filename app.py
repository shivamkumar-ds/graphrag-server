import re
from flask import Flask, request, jsonify
from collections import deque, defaultdict

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Helpers: entity/relation extraction (heuristic, no external LLM available)
# ---------------------------------------------------------------------------

STOPWORDS_SENTSTART = {
    "The", "This", "That", "These", "Those", "It", "They", "He", "She",
    "A", "An", "In", "On", "At", "For", "With", "As", "Later", "Then",
    "After", "Before", "Since", "While", "When", "Also",
}

ORG_SUFFIXES = ("Inc", "Corp", "Corporation", "Labs", "Systems", "Technologies",
                "Technology", "Ventures", "Group", "LLC", "Company", "Co")

FRAMEWORK_HINTS = ("framework", "library", "sdk", "toolkit")
PRODUCT_HINTS = ("product", "tool", "app", "application", "platform", "model")
ORG_HINTS = ("company", "organization", "startup", "firm", "enterprise")

PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]*(?:[\-\.][A-Z0-9][a-zA-Z0-9]*)*(?:\s+[A-Z][a-zA-Z0-9]*(?:[\-\.][A-Z0-9][a-zA-Z0-9]*)*)*)\b"
)


def clean_name(name):
    return name.strip().strip(".,;:").strip()


def extract_candidate_entities(text):
    candidates = {}
    for m in PROPER_NOUN_RE.finditer(text):
        name = clean_name(m.group(1))
        if not name:
            continue
        first_word = name.split()[0]
        if name in STOPWORDS_SENTSTART:
            continue
        if first_word in STOPWORDS_SENTSTART and len(name.split()) == 1:
            continue
        if len(name) < 2:
            continue
        candidates[name] = candidates.get(name, 0) + 1
    names = sorted(candidates.keys(), key=len, reverse=True)
    final = []
    for n in names:
        if any(n != other and (f" {n} " in f" {other} " or other.startswith(n + " ") or other.endswith(" " + n)) for other in final):
            continue
        final.append(n)
    return final


def guess_entity_type(name, text):
    lower_ctx = text.lower()
    idx = lower_ctx.find(name.lower())
    window = lower_ctx[max(0, idx - 30): idx + len(name) + 30]

    def has_word(w, text_window):
        return re.search(rf"\b{re.escape(w)}\b", text_window) is not None

    words = name.split()
    if len(words) == 2 and all(w[0].isupper() for w in words) and not any(
        w.isupper() for w in words
    ):
        return "Person"

    if any(name.endswith(suf) or has_word(suf.lower(), window) for suf in ORG_SUFFIXES):
        return "Organization"
    if any(has_word(h, window) for h in FRAMEWORK_HINTS):
        return "Framework"
    if any(has_word(h, window) for h in ORG_HINTS):
        return "Organization"
    if any(has_word(h, window) for h in PRODUCT_HINTS):
        return "Product"
    if len(words) == 1 and name.isupper():
        return "Organization"
    known_orgs = {"OpenAI", "Google", "Microsoft", "Meta", "Anthropic", "Amazon", "IBM"}
    if name in known_orgs:
        return "Organization"
    known_frameworks = {"LangChain", "TensorFlow", "PyTorch", "React", "Django", "Flask", "Kubernetes"}
    if name in known_frameworks:
        return "Framework"
    return "Product"


def build_patterns():
    PN = r"([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3})"
    DET = r"(?:the\s+|a\s+|an\s+)?"
    patterns = [
        (rf"{PN}\s+(?:was|is)\s+(?:co-founded|founded|created|started|built)\s+by\s+{PN}", "REV_CREATE"),
        (rf"{PN}\s+(?:co-founded|founded|created|started|built)\s+{DET}{PN}", "FWD_CREATE"),
        (rf"{PN}\s+(?:was|is)\s+developed\s+by\s+{PN}", "REV_DEVELOP"),
        (rf"{PN}\s+developed\s+{DET}{PN}", "FWD_DEVELOP"),
        (rf"{PN}\s+integrates?\s+with\s+{DET}{PN}", "FWD_INTEGRATE"),
        (rf"{PN}\s+(?:was|is)\s+integrated\s+into\s+{DET}{PN}", "FWD_INTEGRATE_INTO"),
        (rf"{PN}\s+hired\s+{DET}{PN}", "FWD_HIRE"),
        (rf"{PN}\s+(?:was|is)\s+authored\s+by\s+{PN}", "REV_AUTHOR"),
        (rf"{PN}\s+authored\s+{DET}{PN}", "FWD_AUTHOR"),
        (rf"{PN}\s+wrote\s+{DET}{PN}", "FWD_AUTHOR"),
    ]
    return [(re.compile(p), tag) for p, tag in patterns]


REL_PATTERNS = build_patterns()


def extract_relationships(text, entities_by_name):
    rels = []
    seen = set()
    for regex, tag in REL_PATTERNS:
        for m in regex.finditer(text):
            a = clean_name(m.group(1))
            b = clean_name(m.group(2))
            if a == b:
                continue

            if tag == "REV_CREATE":
                target, source = a, b
                rel_type = "DEVELOPED" if entities_by_name.get(target) in ("Framework", "Product") else "FOUNDED"
                src, tgt = source, target
            elif tag == "FWD_CREATE":
                source, target = a, b
                rel_type = "DEVELOPED" if entities_by_name.get(target) in ("Framework", "Product") else "FOUNDED"
                src, tgt = source, target
            elif tag == "REV_DEVELOP":
                src, tgt, rel_type = b, a, "DEVELOPED"
            elif tag == "FWD_DEVELOP":
                src, tgt, rel_type = a, b, "DEVELOPED"
            elif tag == "FWD_INTEGRATE":
                src, tgt, rel_type = a, b, "INTEGRATED_INTO"
            elif tag == "FWD_INTEGRATE_INTO":
                src, tgt, rel_type = a, b, "INTEGRATED_INTO"
            elif tag == "FWD_HIRE":
                src, tgt, rel_type = a, b, "HIRED"
            elif tag == "REV_AUTHOR":
                src, tgt, rel_type = b, a, "AUTHORED"
            elif tag == "FWD_AUTHOR":
                src, tgt, rel_type = a, b, "AUTHORED"
            else:
                continue

            key = (src, tgt, rel_type)
            if key in seen:
                continue
            seen.add(key)
            rels.append({"source": src, "target": tgt, "relation": rel_type})
    return rels


@app.route("/extract-graph", methods=["POST"])
def extract_graph():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")

    candidate_names = extract_candidate_entities(text)
    entities_by_name = {}
    for name in candidate_names:
        entities_by_name[name] = guess_entity_type(name, text)

    relationships = extract_relationships(text, entities_by_name)

    all_names = set(entities_by_name.keys())
    for r in relationships:
        all_names.add(r["source"])
        all_names.add(r["target"])
        entities_by_name.setdefault(r["source"], guess_entity_type(r["source"], text))
        entities_by_name.setdefault(r["target"], guess_entity_type(r["target"], text))

    entities = [{"name": n, "type": entities_by_name[n]} for n in all_names]

    return jsonify({"entities": entities, "relationships": relationships})


# ---------------------------------------------------------------------------
# /graph-query : multi-hop reasoning over a supplied graph
# ---------------------------------------------------------------------------

RELATION_STEMS = {
    "FOUNDED": ["found", "creat", "start", "co-found"],
    "DEVELOPED": ["develop", "creat", "build", "built"],
    "INTEGRATED_INTO": ["integrat", "connect", "plugin", "compat"],
    "HIRED": ["hir", "employ", "join"],
    "AUTHORED": ["author", "wrote", "writ", "publish"],
    "CREATED": ["creat", "found", "develop", "build", "built"],
}

TYPE_KEYWORDS = {
    "Person": ["who"],
    "Organization": ["company", "organization", "startup", "firm"],
    "Framework": ["framework", "library"],
    "Product": ["product", "tool", "app", "model"],
}


def stem_words(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def question_relation_stems(question):
    words = stem_words(question)
    found_stems = set()
    for rel, stems in RELATION_STEMS.items():
        for s in stems:
            for w in words:
                if w.startswith(s):
                    found_stems.add(s)
    return found_stems


def infer_target_type(question):
    q = question.lower()
    for etype, kws in TYPE_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                return etype
    return None


def find_anchor_entities(question, entities):
    q_lower = question.lower()
    matches = []
    for e in entities:
        name = e.get("name", "")
        if not name:
            continue
        if name.lower() in q_lower:
            matches.append(name)
    matches.sort(key=len, reverse=True)
    final = []
    for m in matches:
        if any(m != f and m.lower() in f.lower() for f in final):
            continue
        final.append(m)
    return final


def edge_matches_stems(relation, stems_of_interest):
    if not stems_of_interest:
        return True
    rel_lower = relation.lower()
    for s in stems_of_interest:
        if s in rel_lower:
            return True
    for s in RELATION_STEMS.get(relation.upper(), []):
        if s in stems_of_interest:
            return True
    return False


def build_adjacency(relationships):
    adj = defaultdict(list)
    for r in relationships:
        s, t, rel = r.get("source"), r.get("target"), r.get("relation", "")
        if s is None or t is None:
            continue
        adj[s].append((t, rel, "fwd"))
        adj[t].append((s, rel, "rev"))
    return adj


def bfs_best_path(start, adj, entity_types, target_type, stems_of_interest, max_hops=4):
    visited = {start}
    queue = deque([(start, [start], set(), [])])
    best = None
    while queue:
        node, path, stems_used, rels_used = queue.popleft()
        if len(path) > 1:
            node_type = entity_types.get(node)
            type_ok = (target_type is None) or (node_type == target_type)
            if type_ok and (not stems_of_interest or stems_used):
                score = (len(path), -len(stems_used))
                if best is None or score < best[0]:
                    best = (score, path, rels_used)
                if target_type is not None and stems_used >= stems_of_interest:
                    return path, rels_used
        if len(path) - 1 >= max_hops:
            continue
        for (neigh, rel, direction) in adj.get(node, []):
            if neigh in path:
                continue
            new_stems = set(stems_used)
            for s in stems_of_interest:
                if s in rel.lower():
                    new_stems.add(s)
            queue.append((neigh, path + [neigh], new_stems, rels_used + [rel]))
    if best:
        return best[1], best[2]
    return None, None


@app.route("/graph-query", methods=["POST"])
def graph_query():
    data = request.get_json(force=True) or {}
    question = data.get("question", "")
    graph = data.get("graph", {}) or {}
    entities = graph.get("entities", [])
    relationships = graph.get("relationships", [])

    entity_types = {e.get("name"): e.get("type") for e in entities if e.get("name")}
    adj = build_adjacency(relationships)

    stems_of_interest = question_relation_stems(question)
    target_type = infer_target_type(question)
    anchors = find_anchor_entities(question, entities)

    best_path = None
    best_rels = None

    if anchors:
        for anchor in anchors:
            path, rels = bfs_best_path(anchor, adj, entity_types, target_type, stems_of_interest)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path, best_rels = path, rels
    else:
        for node in adj.keys():
            path, rels = bfs_best_path(node, adj, entity_types, target_type, stems_of_interest)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path, best_rels = path, rels

    if best_path and len(best_path) > 1:
        answer = best_path[-1]
        return jsonify({
            "answer": answer,
            "reasoning_path": best_path,
            "hops": len(best_path) - 1,
        })

    fallback_answer = None
    if target_type:
        for e in entities:
            if e.get("type") == target_type:
                fallback_answer = e.get("name")
                break
    return jsonify({
        "answer": fallback_answer,
        "reasoning_path": anchors if anchors else [],
        "hops": max(len(anchors) - 1, 0) if anchors else 0,
    })


# ---------------------------------------------------------------------------
# /community-summary : template based natural-language summary
# ---------------------------------------------------------------------------

REL_PHRASES = {
    "FOUNDED": ("founded by {other}", "founded {other}"),
    "CREATED": ("created by {other}", "created {other}"),
    "DEVELOPED": ("developed by {other}", "developed {other}"),
    "INTEGRATED_INTO": ("integrated into {other}", "that integrates with {other}"),
    "HIRED": ("hired by {other}", "hired {other}"),
    "AUTHORED": ("authored by {other}", "authored {other}"),
}

TYPE_ARTICLE = {
    "Framework": "an AI framework",
    "Organization": "an organization",
    "Product": "a product",
    "Person": "a person",
}


def article_for(etype):
    return TYPE_ARTICLE.get(etype, "an entity")


@app.route("/community-summary", methods=["POST"])
def community_summary():
    data = request.get_json(force=True) or {}
    community_id = data.get("community_id", "")
    entity_names = data.get("entities", [])
    relationships = data.get("relationships", [])

    degree = defaultdict(int)
    for r in relationships:
        degree[r.get("source")] += 1
        degree[r.get("target")] += 1

    if entity_names:
        central = max(entity_names, key=lambda n: degree.get(n, 0))
    elif relationships:
        central = max(degree, key=degree.get) if degree else None
    else:
        central = None

    clauses = []
    for r in relationships:
        s, t, rel = r.get("source"), r.get("target"), r.get("relation", "")
        rel_key = rel.upper() if rel else ""
        phrase_pair = REL_PHRASES.get(rel_key, ("related to {other}", "related to {other}"))
        if t == central:
            clauses.append(phrase_pair[0].format(other=s))
        elif s == central:
            clauses.append(phrase_pair[1].format(other=t))

    if central is None:
        summary = f"Community {community_id} has no identifiable central entity."
    else:
        others = [n for n in entity_names if n != central]
        if clauses:
            clause_text = ", ".join(clauses)
            summary = f"This community centers around {central}, {clause_text}."
        elif others:
            summary = f"This community centers around {central}, connected to {', '.join(others)}."
        else:
            summary = f"This community consists solely of {central}."

    return jsonify({"community_id": community_id, "summary": summary})


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "GraphRAG Pipeline"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
