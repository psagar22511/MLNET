using Microsoft.ML;

namespace ClusteringExample
{
    public class ClusteringService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<CustomerData, ClusterPrediction> _predictionEngine;

        private readonly string _modelPath;
        public ClusteringService(string dataPath, string modelPath)
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
                .CreatePredictionEngine<CustomerData, ClusterPrediction>(_model);
        }
        private void TrainAndSaveModel(string dataPath)
        {
            Console.WriteLine("Training model...");

            // Load Data
            //IDataView dataView = _mlContext.Data.LoadFromTextFile<EmailData>(
            //    dataPath,
            //    hasHeader: true,
            //    separatorChar: ',');
            var samples = new List<CustomerData>
            {
                new CustomerData{ Age = 25, AnnualIncome = 50000 },
                new CustomerData{ Age = 45, AnnualIncome = 100000 },
                new CustomerData{ Age = 23, AnnualIncome = 40000 },
                new CustomerData{ Age = 40, AnnualIncome = 90000 },
                new CustomerData{ Age = 36, AnnualIncome = 80000 }
            };
            var data = _mlContext.Data.LoadFromEnumerable(samples);

            // Create Pipeline
            var pipeline = _mlContext.Transforms.Concatenate("Features", new[] { "Age", "AnnualIncome" })
                        .Append(_mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 2));

            // Train Model
            _model = pipeline.Fit(data);

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
        public ClusterPrediction Predict(CustomerData customerData)
        {
            var input = new CustomerData { Age = customerData.Age, AnnualIncome = customerData.AnnualIncome };
            return _predictionEngine.Predict(input);
        }
    }
}
