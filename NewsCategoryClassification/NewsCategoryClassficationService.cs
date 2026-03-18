using Microsoft.ML;

namespace NewsCategoryClassification
{
    public class NewsCategoryClassficationService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        //private PredictionEngine<NewsData, NewsPrediction> _predictionEngine;
        private readonly string _modelPath;

        public NewsCategoryClassficationService(string dataPath, string modelPath)
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
            //_predictionEngine = _mlContext.Model
            //    .CreatePredictionEngine<NewsData, NewsPrediction>(_model);
        }
        private void TrainAndSaveModel(string dataPath)
        {
            Console.WriteLine("Training model...");

            // Load Data
            IDataView dataView = _mlContext.Data.LoadFromTextFile<NewsData>(
                dataPath,
                hasHeader: true,
                separatorChar: ',');

            // Create Pipeline
            var pipeline = _mlContext.Transforms.Conversion
                        .MapValueToKey("Label")
                        .Append(_mlContext.Transforms.Text.FeaturizeText("Features", "Text"))
                        .Append(_mlContext.MulticlassClassification.Trainers
                            .SdcaMaximumEntropy())
                        .Append(_mlContext.Transforms.Conversion
                        .MapKeyToValue("PredictedLabel"));

            // Train Model
            _model = pipeline.Fit(dataView);

            #region Model Evaluation Process
            var predictions = _model.Transform(dataView);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions);
            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"MacroAccuracy: {metrics.PerClassLogLoss:P2}");
            Console.WriteLine($"MacroAccuracy: {metrics.TopKAccuracy:P2}");
            Console.WriteLine($"MacroAccuracy: {metrics.TopKAccuracyForAllK:P2}");
            Console.WriteLine($"MacroAccuracy: {metrics.TopKPredictionCount:P2}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:P2}");
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
    }
}
