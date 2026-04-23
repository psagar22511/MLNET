using System.Reflection;

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

            // Example-1
            var data2 = new List<ClusterItemData2>
            {
                new ClusterItemData2 { Size = 2, Weight = 1, Hardness = 1,  FruitName="Apple" }, // Apple
                new ClusterItemData2 { Size = 3, Weight = 1, Hardness = 1,  FruitName="Banana" }, // Banana
                new ClusterItemData2 { Size = 4, Weight = 3, Hardness = 3,  FruitName="Toy Car" }, // Toy Car
                new ClusterItemData2 { Size = 3, Weight = 2, Hardness = 2,  FruitName="Ball" }  // Ball
            };
            var trainingData2 = _mlContext.Data.LoadFromEnumerable(data2);
            var pipeline2 = _mlContext.Transforms
                        .Concatenate("Features", nameof(ClusterItemData2.Size),
                                                  nameof(ClusterItemData2.Weight),
                                                  nameof(ClusterItemData2.Hardness))
                        .Append(_mlContext.Clustering.Trainers.KMeans(
                                featureColumnName: "Features",
                                numberOfClusters: 2));
            var model2 = pipeline2.Fit(trainingData2);
            var predictor2 = _mlContext.Model.CreatePredictionEngine<ClusterItemData2, ClusterPrediction2>(model2);
            foreach (var item in data2)
            {
                var prediction3 = predictor2.Predict(item);

                Console.WriteLine($"Fruit: {item.FruitName}");
                Console.WriteLine($"Cluster ID: {prediction3.ClusterId}");

                for (int i = 0; i < prediction3?.Distances?.Length; i++)
                {
                    Console.WriteLine($"Distance to Cluster {i}: {prediction3.Distances[i]:0.00}");
                }

                Console.WriteLine("------------------------");
            }

            var testItem = new ClusterItemData2 { Size = 2.5f, Weight = 1f, Hardness = 1f, FruitName = "Mango" };

            var prediction = predictor2.Predict(testItem);

            Console.WriteLine($"Cluster: {prediction.ClusterId}");
                        

            //Example-2
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
