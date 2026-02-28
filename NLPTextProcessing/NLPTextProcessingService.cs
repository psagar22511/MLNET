using Microsoft.ML;
using NLPTextProcessing;

namespace NewsCategoryClassification
{
    public class NLPTextProcessingService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<SentimentData, SentimentPrediction> _predictionEngine;
        private readonly string _modelPath;

        public NLPTextProcessingService(string dataPath, string modelPath)
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
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(_model);
        }
        private void LoadModel()
        {
            using var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read);
            _model = _mlContext.Model.Load(stream, out _);

            Console.WriteLine("Model loaded from file.");
        }
        private void TrainAndSaveModel(string dataPath)
        {
            Console.WriteLine("Training model...");

            //Created Date or Load Data From CSV File
            //Not used "dataPath" - We not load data from CSV.
            var trainingData = new List<SentimentData>
            {
                new SentimentData { Text = "This product is amazing!", Label = true },
                new SentimentData { Text = "I love this!", Label = true },
                new SentimentData { Text = "Worst experience ever.", Label = false },
                new SentimentData { Text = "Terrible service.", Label = false }
            };
            IDataView dataView = _mlContext.Data.LoadFromEnumerable(trainingData);

            // Create Pipeline
            var pipeline = _mlContext.Transforms.Text.FeaturizeText(
                    outputColumnName: "Features",
                    inputColumnName: nameof(SentimentData.Text))
                .Append(_mlContext.BinaryClassification.Trainers
                    .SdcaLogisticRegression());

            /*FeaturizeText() automatically performs:(It converts raw text into numerical feature vectors usable by ML algorithms.)
                Text normalization
                Tokenization
                Stop-word removal
                N-grams extraction
                TF-IDF vectorization */

            // Train model
            _model = pipeline.Fit(dataView);

            // Save model
            _mlContext.Model.Save(_model, dataView.Schema, _modelPath);
        }
        public SentimentPrediction Predict(string text)
        {
            var input = new SentimentData { Text = text };
            return _predictionEngine.Predict(input);
        }
    }
}
