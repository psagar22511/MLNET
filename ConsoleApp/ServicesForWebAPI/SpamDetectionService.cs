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

            // Create Prediction Engine (Predict single sample)
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

            #region Model Evaluation Process
            var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = split.TrainSet; //used to train the model
            var testData = split.TestSet; //used only for evaluation

            var predictions = _model.Transform(testData);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions);

            Console.WriteLine("===== Model Evaluation Metrics =====");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine($"F1 Score: {metrics.LogLoss:P2}");
            Console.WriteLine($"F1 Score: {metrics.LogLossReduction:P2}");
            Console.WriteLine($"F1 Score: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.NegativePrecision:P2}");
            Console.WriteLine($"F1 Score: {metrics.NegativeRecall:P2}");
            Console.WriteLine($"F1 Score: {metrics.PositivePrecision:P2}");
            Console.WriteLine($"F1 Score: {metrics.PositiveRecall:P2}");
            #endregion

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
