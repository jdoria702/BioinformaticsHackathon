class LessonService:
    """
    This class provides topic contenxt for the tutor.
    The information is currently hardcoded and very simple.
    Later on, we'll reroute this to query a vector DB / documents / files.
    """
    _TOPICS = {
        "sequence_alignment": (
            "Sequence alignment is the process of arranging DNA/RNA/protein sequences "
            "to identify regions of similarity. Common types: global (Needleman-Wunsch) "
            " and local (Smith-Waterman). Key concepts include scoring matrices, gap "
            "penalties, and dynamic programming."
        ),
        "blast": (
            "BLAST is a heuristic algorithm to find local similarities between sequences. "
            "It uses seed-and-extend, reports hits with E-values, bit scores, and alignments."
        ),
    }

    # Function that allows agent to retrieve topics:
    def get_topic_context(self, topic: str) -> str:
        return self._TOPICS.get(
            topic,
            "General bioinformatics tutoring context. Ask the user clarifying questions and explain step-by-step."
        )