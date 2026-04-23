using Microsoft.ML;

namespace RankingExample
{
    public class RankingService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<RankingData, RankingPrediction> _predictionEngine;

        private readonly string _modelPath;
        public RankingService(string dataPath, string modelPath)
        {
            _mlContext = new MLContext();
            _modelPath = modelPath;

            if (File.Exists(_modelPath))
            {
                //LoadModel();
                File.Delete(_modelPath);
            }
            else
            {
                TrainAndSaveModel(dataPath);
            }

            // Create Prediction Engine
            _predictionEngine = _mlContext.Model
                .CreatePredictionEngine<RankingData, RankingPrediction>(_model);
        }
        private void TrainAndSaveModel(string dataPath)
        {
            Console.WriteLine("Training model...");

            // Load Data
            var samples = new List<RankingData>()
            {
                new RankingData { Label = 3, Feature1 = 1, Feature2 = 0.1f, GroupId = 1 },
                new RankingData { Label = 2, Feature1 = 2, Feature2 = 0.2f, GroupId = 1 },
                new RankingData { Label = 1, Feature1 = 3, Feature2 = 0.3f, GroupId = 1 },
                new RankingData { Label = 0, Feature1 = 4, Feature2 = 0.4f, GroupId = 1 },

                new RankingData { Label = 3, Feature1 = 1, Feature2 = 0.5f, GroupId = 2 },
                new RankingData { Label = 2, Feature1 = 2, Feature2 = 0.6f, GroupId = 2 },
                new RankingData { Label = 1, Feature1 = 3, Feature2 = 0.7f, GroupId = 2 },
                new RankingData { Label = 0, Feature1 = 4, Feature2 = 0.8f, GroupId = 2 },

                new RankingData { Label = 3, Feature1 = 1, Feature2 = 0.9f, GroupId = 3 },
                new RankingData { Label = 2, Feature1 = 2, Feature2 = 1.0f, GroupId = 3 },
                new RankingData { Label = 1, Feature1 = 3, Feature2 = 1.1f, GroupId = 3 },
                new RankingData { Label = 0, Feature1 = 4, Feature2 = 1.2f, GroupId = 3 },
            };
            //var mlContext = new MLContext();

            var data = _mlContext.Data.LoadFromEnumerable(samples);

            // Create Pipeline
            var pipeline =
                _mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "GroupIdKey",
                    inputColumnName: nameof(RankingData.GroupId))

                .Append(_mlContext.Transforms.Concatenate(
                    "Features",
                    nameof(RankingData.Feature1),
                    nameof(RankingData.Feature2)))

                .Append(_mlContext.Ranking.Trainers.LightGbm(
                    new Microsoft.ML.Trainers.LightGbm.LightGbmRankingTrainer.Options
                    {
                        LabelColumnName = "Label",
                        FeatureColumnName = "Features",
                        RowGroupColumnName = "GroupIdKey",
                        MinimumExampleCountPerGroup = 1,
                        NumberOfLeaves = 10,
                        MinimumExampleCountPerLeaf = 1,
                        LearningRate = 0.1
                    }));

            Console.WriteLine(data.Schema);

            // Train Model
            _model = pipeline.Fit(data);

            // Save model
            _mlContext.Model.Save(_model, data.Schema, _modelPath);

            Console.WriteLine("Model trained and saved.");
        }
        public RankingPrediction Predict(RankingData rankingData)
        {
            var input = new RankingData { Feature1 = rankingData.Feature1, Feature2 = rankingData.Feature2, GroupId = rankingData.GroupId };
            return _predictionEngine.Predict(input);
        }
        public IEnumerable<RankingPrediction> PredictBatch(IEnumerable<RankingData> data)
        {
            var dataView = _mlContext.Data.LoadFromEnumerable(data);

            var transformed = _model.Transform(dataView);

            return _mlContext.Data
                .CreateEnumerable<RankingPrediction>(transformed, reuseRowObject: false);
        }
    }
}
