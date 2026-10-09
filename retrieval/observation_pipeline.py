from retrieval.attribute_verifier import AttributeVerifier
from retrieval.event_builder import EventBuilder
from retrieval.index_builder import ObservationIndexBuilder
from retrieval.observation import Observation
from retrieval.observation_encoder import ObservationEncoder
from retrieval.pipeline import RetrievalPipeline
from retrieval.retrieval_result import RetrievalResult
from retrieval.result_builder import ResultBuilder


class ObservationRetrievalPipeline:
    """
    Complete observation-to-event retrieval pipeline.

    Flow:

        Person 1 observations
                ↓
        Object crop embeddings
                ↓
        FAISS
                ↓
        Natural-language query
                ↓
        Query parsing + filtering
                ↓
        Semantic ranking
                ↓
        Temporal event grouping
                ↓
        Attribute verification
                ↓
        RetrievalResult
    """

    def __init__(
        self,
        encoder: ObservationEncoder,
        retrieval_pipeline: RetrievalPipeline,
        max_gap: float = 2.0,
    ):
        self.index_builder = ObservationIndexBuilder(
            encoder=encoder,
            embedding_dimension=512,
        )

        self.retrieval_pipeline = retrieval_pipeline

        self.event_builder = EventBuilder(
            max_gap=max_gap,
        )

        self.result_builder = ResultBuilder()
        self.attribute_verifier = AttributeVerifier()

    def index_observations(
        self,
        observations: list[Observation],
    ) -> None:
        """
        Encode and index Person 1 observations.
        """

        self.index_builder.add_observations(
            observations
        )

        self.retrieval_pipeline.retrieval_engine.vector_index = (
            self.index_builder.vector_index
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """
        Search indexed observations and return temporal
        retrieval results with attribute verification.
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        parsed_query = self.retrieval_pipeline.query_parser.parse(query)
        requested_attributes = parsed_query.get("attributes", [])
        color_attributes = [
            a for a in requested_attributes
            if a in self.attribute_verifier.supported_attributes()
        ]

        if color_attributes:
            pool_size = (
                candidate_pool_size
                if candidate_pool_size is not None
                else max(top_k * 25, 150)
            )
        else:
            pool_size = (
                candidate_pool_size
                if candidate_pool_size is not None
                else max(top_k * 10, 50)
            )

        candidates = self.retrieval_pipeline.search(
            query=query,
            top_k=pool_size,
            filters=filters,
        )

        events = self.event_builder.build_events(
            candidates
        )

        if color_attributes:
            verified_events = []
            for event in events:
                event_obs = event.get("observations", [])
                if not event_obs:
                    continue

                all_attrs_verified = True
                matched_details = {}
                best_verified_obs = None
                max_frac = -1.0

                for attr in color_attributes:
                    attr_verified = False
                    for obs in event_obs:
                        ver = self.attribute_verifier.verify_crop(obs.get("crop_path"), attr)
                        if ver["verified"]:
                            attr_verified = True
                            if ver["fraction"] > max_frac:
                                max_frac = ver["fraction"]
                                best_verified_obs = obs
                            matched_details[attr] = ver
                            break
                    if not attr_verified:
                        all_attrs_verified = False
                        break

                if all_attrs_verified:
                    event["verification_status"] = "attribute_verified"
                    event["attribute_details"] = matched_details
                    if best_verified_obs:
                        # Ensure best observation chosen by ResultBuilder is the verified observation
                        event["observations"] = [best_verified_obs] + [o for o in event_obs if o != best_verified_obs]
                        event["best_timestamp"] = best_verified_obs.get("timestamp", event["best_timestamp"])
                    verified_events.append(event)

            events = verified_events
        else:
            for event in events:
                event["verification_status"] = "visual_similarity"
                event["attribute_details"] = None

        events.sort(
            key=lambda event: event.get("best_score", 0.0),
            reverse=True,
        )

        results = self.result_builder.build(
            events
        )

        return results[:top_k]

    def __len__(self) -> int:
        return len(self.index_builder)