using Microsoft.AspNetCore.Mvc;

using MLAPI.Model;
using MLAPI.ServicesForWebAPI;

namespace MLAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class SpamDetectionController : ControllerBase
    {
        private readonly SpamDetectionService _spamDetectionService;
        public SpamDetectionController(SpamDetectionService spamDetectionService)
        {
            _spamDetectionService = spamDetectionService;
        }
        [HttpPost("predict")]
        public IActionResult Predict([FromBody] EmailData request)
        {
            //POST MAN = https://localhost:7278/api/SpamDetection/predict
            //Body - JSON - {"Text" : "How are you?" }
            var result = _spamDetectionService.Predict(request.Text);

            return Ok(new
            {
                Prediction = result.PredictedLabel ? "Spam" : "Not Spam",
                Probability = result.Probability
            });
        }
    }
}
