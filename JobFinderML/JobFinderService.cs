using Microsoft.ML;

namespace JobFinderML
{
    public class JobFinderService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private readonly string _modelPath;
        private PredictionEngine<JobData, JobPrediction> _predictionEngine;

        public JobFinderService(string dataPath, string modelPath)
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
                .CreatePredictionEngine<JobData, JobPrediction>(_model);
        }
        private void TrainAndSaveModel(string dataPath)
        {
            Console.WriteLine("Training model...");

            // Load Data
            var data = _mlContext.Data.LoadFromTextFile<JobData>(
                dataPath, hasHeader: true, separatorChar: ',');

            // Create Pipeline
            var pipeline = _mlContext.Transforms.Text.FeaturizeText(
                    "Features", nameof(JobData.Description))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: nameof(JobData.IsRelevant),
                    featureColumnName: "Features"));

            // Train Model
            _model = pipeline.Fit(data);

            //#region Model Evaluation Process
            //var predictions = _model.Transform(dataView);
            //var metrics = _mlContext.MulticlassClassification.Evaluate(predictions);
            //Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
            //Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");
            //Console.WriteLine($"PerClassLogLoss: {metrics.PerClassLogLoss:P2}");
            //Console.WriteLine($"TopKAccuracy: {metrics.TopKAccuracy:P2}");
            //Console.WriteLine($"TopKAccuracyForAllK: {metrics.TopKAccuracyForAllK:P2}");
            //Console.WriteLine($"TopKPredictionCount: {metrics.TopKPredictionCount:P2}");
            //Console.WriteLine($"LogLoss: {metrics.LogLoss:P2}");
            //#endregion

            // Save model
            _mlContext.Model.Save(_model, data.Schema, _modelPath);

            Console.WriteLine("Model trained and saved.");
        }
        private void LoadModel()
        {
            using var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read);
            _model = _mlContext.Model.Load(stream, out _);

            Console.WriteLine("Model loaded from file.");
        }
        public JobPrediction Predict(string title, string desc)
        {
            var input = new JobData { Title = title, Description  = desc};
            return _predictionEngine.Predict(input);
        }
    }
}
