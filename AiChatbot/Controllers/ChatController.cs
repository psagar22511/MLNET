using AiChatbot.Services;

using Microsoft.AspNetCore.Mvc;

namespace AiChatbot.Controllers
{
    [ApiController]
    [Route("chat")]
    public class ChatController : ControllerBase
    {
        private readonly OpenAiService _aiService;

        public ChatController(OpenAiService aiService)
        {
            _aiService = aiService;
        }

        [HttpPost]
        public async Task<IActionResult> Chat([FromBody] ChatRequest request)
        {
            var response = await _aiService.GetResponse(request.Message);
            return Ok(new { reply = response });
        }
    }

    public class ChatRequest
    {
        public string Message { get; set; }
    }
}
