using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

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
                LoadModel();
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
                new RankingData { Label = 3, Feature1 = 1, Feature2 = 0.1f, GroupId =   1 },
                new RankingData { Label = 2, Feature1 = 2, Feature2 = 0.2f, GroupId =   1 },
                new RankingData { Label = 1, Feature1 = 3, Feature2 = 0.3f, GroupId =   1 },
                new RankingData { Label = 3, Feature1 = 1, Feature2 = 0.5f, GroupId =   2 },
                new RankingData { Label = 2, Feature1 = 2, Feature2 = 0.6f, GroupId =   2 },
            };
            var mlContext = new MLContext();

            var data = mlContext.Data.LoadFromEnumerable(samples);

            // Create Pipeline
            var pipeline =
                mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "GroupIdKey",
                    inputColumnName: "GroupId")

                .Append(mlContext.Transforms.Concatenate(
                    "Features",
                    nameof(RankingData.Feature1),
                    nameof(RankingData.Feature2)))

                .Append(mlContext.Ranking.Trainers.LightGbm(
                    labelColumnName: "Label",
                    featureColumnName: "Features",
                    rowGroupColumnName: "GroupIdKey"));

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
        public RankingPrediction Predict(RankingData rankingData)
        {
            var input = new RankingData { Feature1 = rankingData.Feature1, Feature2 = rankingData.Feature2, GroupId = rankingData.GroupId };
            return _predictionEngine.Predict(input);
        }
    }
}
