"""Physical-only OpenAI image-detail compatibility for original high-resolution inputs."""

from providers.openai_vlm import OpenAIProvider


class PhysicalDirectOpenAIProvider(OpenAIProvider):
    @property
    def provider_config(self):
        return {**super().provider_config, "image_detail": "high",
                "image_bytes": "unchanged_original",
                "image_detail_reason": "Original 48MP input exceeds the API 30000-patch rejection limit",
                "image_detail_documentation": "https://developers.openai.com/api/docs/guides/images-vision"}

    def build_payload(self, request):
        payload = super().build_payload(request)
        for content in payload["input"][0]["content"]:
            if content["type"] == "input_image":
                content["detail"] = "high"
        return payload
