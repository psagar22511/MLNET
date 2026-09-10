using dotnet_api.Model;
using dotnet_api.Service;

using Microsoft.AspNetCore.Mvc;

namespace dotnet_api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class PredictionController : Controller
    {
        private readonly MlService _mlService;

        public PredictionController(MlService mlService)
        {
            _mlService = mlService;
        }

        [HttpPost]
        public async Task<IActionResult> Prediction([FromBody] PredictionRequest request)
        {
            var result = await _mlService.GetPrediction(request);
            //return Ok(result);
            return Content(result, "application/json");
        }
    }
}
