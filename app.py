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

# Generic acronyms/terms that are too common as standalone "entities" and
# almost always false positives when captured alone (e.g. "AI" inside
# "AI research organization" or "AI safety").
GENERIC_TERMS = {"AI", "ML", "API", "US", "UK", "CEO", "CTO", "IT", "NLP", "IPO"}

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
        if len(name.split()) == 1 and name.upper() in GENERIC_TERMS:
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


# Trigger verbs -> canonical relation label (literal, matches the spec's
# example which uses "CREATED" verbatim rather than remapping it).
VERB_RELATION = {
    "co-founded": "FOUNDED",
    "cofounded": "FOUNDED",
    "founded": "FOUNDED",
    "co-founder of": "FOUNDED",
    "founder of": "FOUNDED",
    "created": "CREATED",
    "creator of": "CREATED",
    "started": "FOUNDED",
    "built": "DEVELOPED",
    "developed": "DEVELOPED",
    "developer of": "DEVELOPED",
    "integrates": "INTEGRATED_INTO",
    "integrate": "INTEGRATED_INTO",
    "integrated": "INTEGRATED_INTO",
    "hired": "HIRED",
    "authored": "AUTHORED",
    "author of": "AUTHORED",
    "wrote": "AUTHORED",
}

# Ordered longest-first so multi-word triggers ("co-founder of") are tried
# before shorter overlapping ones.
VERB_TRIGGERS = sorted(VERB_RELATION.keys(), key=len, reverse=True)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CONJ_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+")


def find_entity_mentions(sentence, entity_names):
    """Return list of (name, start, end) for every entity mention in the
    sentence, longest names matched first so overlaps favor specificity."""
    mentions = []
    lower_sentence = sentence.lower()
    ordered = sorted(entity_names, key=len, reverse=True)
    taken = [False] * len(sentence)
    for name in ordered:
        name_lower = name.lower()
        start = 0
        while True:
            idx = lower_sentence.find(name_lower, start)
            if idx == -1:
                break
            end = idx + len(name_lower)
            if not any(taken[idx:end]):
                mentions.append((name, idx, end))
                for i in range(idx, end):
                    taken[i] = True
            start = idx + 1
    mentions.sort(key=lambda m: m[1])
    return mentions


def split_conjunction(sentence_slice, entity_names_set):
    """Split a phrase like 'Elon Musk and Sam Altman' into individual entity
    names, only keeping pieces that are known entities."""
    parts = [p.strip() for p in CONJ_SPLIT_RE.split(sentence_slice) if p.strip()]
    return [p for p in parts if p in entity_names_set]


def extract_relationships(text, entities_by_name):
    entity_names = list(entities_by_name.keys())
    entity_names_set = set(entity_names)
    rels = []
    seen = set()

    sentences = SENTENCE_SPLIT_RE.split(text)
    for sentence in sentences:
        mentions = find_entity_mentions(sentence, entity_names)
        if len(mentions) < 2:
            continue

        lower_sentence = sentence.lower()
        # Find the first (leftmost) relation-trigger verb in this sentence.
        best_trigger = None
        best_pos = None
        for trigger in VERB_TRIGGERS:
            pos = lower_sentence.find(trigger)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos = pos
                best_trigger = trigger
        if best_trigger is None:
            continue

        rel_type = VERB_RELATION[best_trigger]
        verb_start = best_pos
        verb_end = best_pos + len(best_trigger)

        before_mentions = [m for m in mentions if m[2] <= verb_start]
        after_mentions = [m for m in mentions if m[1] >= verb_end]

        if not before_mentions or not after_mentions:
            continue

        # Detect passive voice: "<object> was/is <verb> by <subject>"
        pre_verb_text = lower_sentence[:verb_start]
        is_passive = bool(re.search(r"\b(was|is|were|are)\s+$", pre_verb_text.rstrip() + " ")) or \
            bool(re.search(r"\b(was|is|were|are)\s+\w*\s*$", pre_verb_text))
        post_verb_text = lower_sentence[verb_end:verb_end + 10]
        has_by = post_verb_text.strip().startswith("by")

        # Expand the nearest mention group on each side to catch conjunctions
        # like "Elon Musk and Sam Altman" that were split into separate
        # mentions but sit right next to each other before the verb.
        left_names = [m[0] for m in before_mentions]
        right_names = [m[0] for m in after_mentions]

        if is_passive and has_by:
            # object(s) before verb, subject(s) after "by"
            objects = left_names
            subjects = right_names
        else:
            subjects = left_names
            objects = right_names

        # Special-case three-argument integration phrasing like
        # "Microsoft integrated GPT-4 into Copilot" — the actor (Microsoft)
        # doing the integrating is not itself part of the INTEGRATED_INTO
        # relation; the real relation is GPT-4 -> Copilot.
        if rel_type == "INTEGRATED_INTO":
            prep_match = re.search(r"\b(into|with)\b", lower_sentence[verb_end:verb_end + 40])
            if prep_match:
                prep_pos = verb_end + prep_match.start()
                prep_end = verb_end + prep_match.end()
                middle_names = [m[0] for m in mentions if m[1] >= verb_end and m[2] <= prep_pos]
                after_prep_names = [m[0] for m in mentions if m[1] >= prep_end]
                if middle_names and after_prep_names:
                    subjects = middle_names
                    objects = after_prep_names
                elif after_prep_names:
                    subjects = left_names
                    objects = after_prep_names

        for subj in subjects:
            for obj in objects:
                if subj == obj:
                    continue
                key = (subj, obj, rel_type)
                if key in seen:
                    continue
                seen.add(key)
                rels.append({"source": subj, "target": obj, "relation": rel_type})

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
