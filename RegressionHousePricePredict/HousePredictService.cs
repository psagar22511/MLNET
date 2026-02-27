using Microsoft.ML;

namespace RegressionHousePricePredict
{
    public class HousePredictService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<HouseData, HousePrediction> _predictionEngine;
        private readonly string _modelPath;
        public HousePredictService(string dataPath, string modelPath)
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
                .CreatePredictionEngine<HouseData, HousePrediction>(_model);
        }
        private void TrainAndSaveModel(string dataPath)
        {
            Console.WriteLine("Training model...");

            // Load Data
            IDataView dataView = _mlContext.Data.LoadFromTextFile<HouseData>(
                dataPath,
                hasHeader: true,
                separatorChar: ',');

            // Create Pipeline
            var pipeline = _mlContext.Transforms.Concatenate("Features",
                        nameof(HouseData.Size),
                        nameof(HouseData.Bedrooms),
                        nameof(HouseData.Age))
                    .Append(_mlContext.Regression.Trainers.Sdca());

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
        public HousePrediction Predict(float size, float bedroom, float age)
        {
            var input = new HouseData { Size = size, Bedrooms = bedroom, Age = age };
            return _predictionEngine.Predict(input);
        }
    }
}
