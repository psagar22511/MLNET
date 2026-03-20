using Microsoft.ML;

namespace AnomalyDetectionExample
{
    public class AnomalyDetectionService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<TimeSeriesData, AnomalyPrediction> _predictionEngine;

        private readonly string _modelPath;
        public AnomalyDetectionService(string dataPath, string modelPath)
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
        }
        private void TrainAndSaveModel(string dataPath)
        {
            Console.WriteLine("Training model...");

            // Load Data
            var data = new List<TimeSeriesData>()
            {
                new TimeSeriesData { Value = 10 },
                new TimeSeriesData { Value = 12 },
                new TimeSeriesData { Value = 11 },
                new TimeSeriesData { Value = 13 },
                new TimeSeriesData { Value = 12 },
                new TimeSeriesData { Value = 1000 }, // anomaly
                new TimeSeriesData { Value = 11 },
                new TimeSeriesData { Value = 10 }
            };
            var mlContext = new MLContext();
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // Create Pipeline
            var pipeline = _mlContext.Transforms.DetectIidSpike(
                    outputColumnName: "Prediction",
                    inputColumnName: "Value",
                    confidence: 95,
                    pvalueHistoryLength: 5);

            // Train Model
            _model = pipeline.Fit(dataView);

            //Detect Anomalies
            var transformedData = _model.Transform(dataView);

            var predictions = mlContext.Data
                            .CreateEnumerable<AnomalyPrediction>(
                                transformedData, reuseRowObject: false);

            int index = 0;

            foreach (var p in predictions)
            {
                Console.WriteLine(
                    $"Index {index++} Alert: {p.Prediction[0]} Score: {p.Prediction[1]}");
            }

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
    }
}
