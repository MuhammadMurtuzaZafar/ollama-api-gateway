"""
Models endpoint handler.
Handles /api/models requests for listing available models.
"""

from collections import List
from time import now

from ..models.response import ModelInfo, ModelsResponse, ErrorResponse
from ..inference.max_engine import get_inference_engine


fn handle_list_models() -> ModelsHandlerResult:
    """
    Handle GET /api/models endpoint.
    Returns list of available models.
    """
    var engine = get_inference_engine()
    if not engine:
        return ModelsHandlerResult.error(
            ErrorResponse(error="Inference engine not initialized", code=503),
            503
        )

    var model_names = engine[0].list_models()
    var models = List[ModelInfo]()

    # Create ModelInfo for each available model
    for i in range(len(model_names)):
        var name = model_names[i]
        var info = ModelInfo(
            name=name,
            size=get_model_size(name),
            modified_at=get_timestamp_string(),
            digest="sha256:" + get_mock_digest(name)
        )
        models.append(info)

    var response = ModelsResponse(models=models)
    return ModelsHandlerResult.success(response.to_json())


fn handle_model_info(model_name: String) -> ModelsHandlerResult:
    """
    Handle GET /api/models/{name} endpoint.
    Returns detailed information about a specific model.
    """
    var engine = get_inference_engine()
    if not engine:
        return ModelsHandlerResult.error(
            ErrorResponse(error="Inference engine not initialized", code=503),
            503
        )

    var model_names = engine[0].list_models()

    # Check if model exists
    var found = False
    for i in range(len(model_names)):
        if model_names[i] == model_name:
            found = True
            break

    if not found:
        return ModelsHandlerResult.error(
            ErrorResponse(error="Model not found: " + model_name, code=404),
            404
        )

    # Return detailed model info
    var info = get_detailed_model_info(model_name)
    return ModelsHandlerResult.success(info)


fn get_detailed_model_info(model_name: String) -> String:
    """Get detailed information about a model."""
    var size = get_model_size(model_name)
    var params = get_model_params(model_name)

    return String(
        '{"name":"' + model_name + '",'
        + '"size":' + String(size) + ','
        + '"modified_at":"' + get_timestamp_string() + '",'
        + '"digest":"sha256:' + get_mock_digest(model_name) + '",'
        + '"details":{'
        + '"format":"gguf",'
        + '"family":"' + get_model_family(model_name) + '",'
        + '"families":["' + get_model_family(model_name) + '"],'
        + '"parameter_size":"' + params + '",'
        + '"quantization_level":"Q4_0"'
        + '},'
        + '"model_info":{'
        + '"general.architecture":"' + get_model_family(model_name) + '",'
        + '"general.parameter_count":' + String(get_param_count(model_name)) + ','
        + '"general.quantization_version":2'
        + '}}'
    )


fn get_model_size(model_name: String) -> Int:
    """Get estimated model size in bytes."""
    if "70b" in model_name:
        return 40_000_000_000  # ~40GB for 70B model
    elif "34b" in model_name:
        return 20_000_000_000  # ~20GB for 34B model
    elif "13b" in model_name:
        return 8_000_000_000   # ~8GB for 13B model
    elif "8b" in model_name:
        return 5_000_000_000   # ~5GB for 8B model
    elif "7b" in model_name:
        return 4_000_000_000   # ~4GB for 7B model
    else:
        return 4_000_000_000   # Default ~4GB


fn get_model_params(model_name: String) -> String:
    """Get model parameter size string."""
    if "70b" in model_name:
        return "70B"
    elif "34b" in model_name:
        return "34B"
    elif "13b" in model_name:
        return "13B"
    elif "8b" in model_name:
        return "8B"
    elif "7b" in model_name:
        return "7B"
    else:
        return "7B"


fn get_param_count(model_name: String) -> Int:
    """Get model parameter count."""
    if "70b" in model_name:
        return 70_000_000_000
    elif "34b" in model_name:
        return 34_000_000_000
    elif "13b" in model_name:
        return 13_000_000_000
    elif "8b" in model_name:
        return 8_000_000_000
    elif "7b" in model_name:
        return 7_000_000_000
    else:
        return 7_000_000_000


fn get_model_family(model_name: String) -> String:
    """Get model family name."""
    if "llama" in model_name:
        return "llama"
    elif "mistral" in model_name:
        return "mistral"
    elif "codellama" in model_name:
        return "llama"
    elif "gemma" in model_name:
        return "gemma"
    else:
        return "unknown"


fn get_mock_digest(model_name: String) -> String:
    """Generate a mock digest hash for a model."""
    # In production, this would be the actual model file hash
    var hash_base = "a1b2c3d4e5f6"
    for i in range(len(model_name)):
        hash_base += String(ord(model_name[i]) % 16)
    return hash_base[:64]


fn get_timestamp_string() -> String:
    """Get current timestamp as ISO string."""
    # In production, format as proper ISO 8601
    var ts = now() // 1_000_000_000
    return String(ts) + "T00:00:00.000000000Z"


@value
struct ModelsHandlerResult:
    """Result of models handler execution."""
    var success: Bool
    var response_body: String
    var status_code: Int

    @staticmethod
    fn success(body: String) -> Self:
        return Self(success=True, response_body=body, status_code=200)

    @staticmethod
    fn error(err: ErrorResponse, code: Int) -> Self:
        return Self(success=False, response_body=err.to_json(), status_code=code)


# OpenAI-compatible /v1/models endpoint
fn handle_openai_models() -> ModelsHandlerResult:
    """
    Handle GET /v1/models endpoint.
    OpenAI-compatible models list.
    """
    var engine = get_inference_engine()
    if not engine:
        return ModelsHandlerResult.error(
            ErrorResponse(error="Inference engine not initialized", code=503),
            503
        )

    var model_names = engine[0].list_models()

    # Build OpenAI-style response
    var models_json = String("[")
    for i in range(len(model_names)):
        if i > 0:
            models_json += ","
        var name = model_names[i]
        models_json += String(
            '{"id":"' + name + '",'
            + '"object":"model",'
            + '"created":' + String(now() // 1_000_000_000) + ','
            + '"owned_by":"modular"}'
        )
    models_json += "]"

    return ModelsHandlerResult.success(
        '{"object":"list","data":' + models_json + '}'
    )
