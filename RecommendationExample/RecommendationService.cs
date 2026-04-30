using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace RecommendationExample
{
    public class RecommendationService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<ProductRating, ProductPrediction> _predictionEngine;
        private readonly string _modelPath;

        public RecommendationService(string dataPath, string modelPath)
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
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<ProductRating, ProductPrediction>(_model);
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
            var trainingData = new List<ProductRating>()
            {
                new ProductRating{userId=1, productId=1, Label=5},
                new ProductRating{userId=1, productId=2, Label=3},
                new ProductRating{userId=1, productId=3, Label=4},

                new ProductRating{userId=2, productId=1, Label=3},
                new ProductRating{userId=2, productId=2, Label=4},

                new ProductRating{userId=3, productId=2, Label=5},
                new ProductRating{userId=3, productId=3, Label=4},
            };
            IDataView dataView = _mlContext.Data.LoadFromEnumerable(trainingData);

            //Configure Matrix Factorization Trainer
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userId",
                MatrixRowIndexColumnName = "productId",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            // Create Pipeline (BOTH BELOW STTEMENT WORKS)
            //var pipeline = _mlContext.Recommendation().Trainers.MatrixFactorization(options);
            var pipeline = _mlContext.Recommendation().Trainers.MatrixFactorization(
                        labelColumnName: nameof(ProductRating.Label),
                        matrixColumnIndexColumnName: nameof(ProductRating.userId),
                        matrixRowIndexColumnName: nameof(ProductRating.productId),
                        numberOfIterations: 20,
                        approximationRank: 100);

            // Train model
            _model = pipeline.Fit(dataView);

            // Save model
            _mlContext.Model.Save(_model, dataView.Schema, _modelPath);
        }
        public ProductPrediction Predict(ProductRating productRating)
        {
            var input = new ProductRating { userId = productRating.userId, productId = productRating.productId };
            return _predictionEngine.Predict(input);
        }
    }
}
