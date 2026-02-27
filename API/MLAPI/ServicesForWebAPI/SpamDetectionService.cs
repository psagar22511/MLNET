using Microsoft.ML;

using MLAPI.Model;

namespace MLAPI.ServicesForWebAPI
{
    public class SpamDetectionService
    {
        private readonly MLContext _mlContext;
        private readonly PredictionEngine<EmailData, Prediction> _predictionEngine;

        public SpamDetectionService(IWebHostEnvironment env)
        {
            _mlContext = new MLContext();

            string modelPath = Path.Combine(env.ContentRootPath, "spamDetectionModel.zip");

            using var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read);

            var model = _mlContext.Model.Load(stream, out _);

            _predictionEngine = _mlContext.Model
                .CreatePredictionEngine<EmailData, Prediction>(model);
        }

        public Prediction Predict(string text)
        {
            return _predictionEngine.Predict(new EmailData { Text = text });
        }
    }
}
