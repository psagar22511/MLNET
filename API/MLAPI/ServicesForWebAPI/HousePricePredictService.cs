using Microsoft.ML;
using MLAPI.Model;

namespace MLAPI.ServicesForWebAPI
{
    public class HousePricePredictService
    {
        private readonly MLContext _mlContext;
        private readonly PredictionEngine<HouseData, HousePrediction> _predictionEngine;

        public HousePricePredictService(IWebHostEnvironment env)
        {
            _mlContext = new MLContext();

            string modelPath = Path.Combine(env.ContentRootPath, "housePricePredict.zip");

            using var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read);

            var model = _mlContext.Model.Load(stream, out _);

            _predictionEngine = _mlContext.Model
                .CreatePredictionEngine<HouseData, HousePrediction>(model);
        }

        public HousePrediction Predict(float size, float bedroom, float age)
        {
            var input = new HouseData { Size = size, Bedrooms = bedroom, Age = age };
            return _predictionEngine.Predict(input);
        }
    }
}
