using Microsoft.ML;

using MLAPI.Model;

namespace ConsoleApp.Services
{
    public class SpamDetectionService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<EmailData, Prediction> _predictionEngine;

        private readonly string _modelPath;
        public SpamDetectionService(string dataPath, string modelPath)
        {
            _mlContext = new MLContext();
            _modelPath = modelPath;

            if (File.Exists(_modelPath))
            {
                LoadModel();
            }
            else
            {
                TrainAndSaveModel(dataPath);
            }

            // Create Prediction Engine
            _predictionEngine = _mlContext.Model
                .CreatePredictionEngine<EmailData, Prediction>(_model);
        }
        private void TrainAndSaveModel(string dataPath)
        {
            Console.WriteLine("Training model...");

            // Load Data
            IDataView dataView = _mlContext.Data.LoadFromTextFile<EmailData>(
                dataPath,
                hasHeader: true,
                separatorChar: ',');

            // Create Pipeline
            var pipeline = _mlContext.Transforms.Text
                .FeaturizeText("Features", nameof(EmailData.Text))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

            // Train Model
            _model = pipeline.Fit(dataView);

            // Save model
            _mlContext.Model.Save(_model, dataView.Schema, _modelPath);

            Console.WriteLine("Model trained and saved.");
        }
        private void LoadModel()
        {
            using var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read);
            _model = _mlContext.Model.Load(stream, out _);

            Console.WriteLine("Model loaded from file.");
        }
        public Prediction Predict(string text)
        {
            var input = new EmailData { Text = text };
            return _predictionEngine.Predict(input);
        }
    }
}
