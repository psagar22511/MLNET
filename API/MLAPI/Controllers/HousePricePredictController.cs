using Microsoft.AspNetCore.Mvc;
using MLAPI.Model;
using MLAPI.ServicesForWebAPI;

namespace MLAPI.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class HousePricePredictController : ControllerBase
    {
        private readonly HousePricePredictService _housePricePredictService;
        public HousePricePredictController(HousePricePredictService housePricePredictService)
        {
            _housePricePredictService = housePricePredictService;
        }
        [HttpPost("predict")]
        public IActionResult Predict([FromBody] HouseData request)
        {
            //POST MAN = https://localhost:7278/api/HousePricePredict/predict
            //Body - JSON - {"Size" : 1500, "Bedrooms" : 3", "Age" : 5 }
            var result = _housePricePredictService.Predict(request.Size, request.Bedrooms, request.Age);

            return Ok(new
            {
                PricePrediction = result.PredictedPrice
            });
        }
    }
}
