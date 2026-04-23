using RankingExample;

class Program
{
    static void Main(string[] args)
    {
        //Ranking Example
        string dataPath = @"C:\SagarPatel\Practice\ML.NET\RankingExample\spamdata.csv";
        string modelPath = @"C:\SagarPatel\Practice\ML.NET\RankingExample\rankingModel.zip";

        var spamDetector = new RankingService(dataPath, modelPath);

        var testData = new List<RankingData>
        {
            new RankingData { Feature1 = 2, Feature2 = 0.15f, GroupId = 1 },
            new RankingData { Feature1 = 3, Feature2 = 0.25f, GroupId = 2 },
            new RankingData { Feature1 = 1, Feature2 = 0.05f, GroupId = 3 },
            new RankingData { Feature1 = 4, Feature2 = 0.45f, GroupId = 4 },
        };

        var results = spamDetector.PredictBatch(testData);

        foreach (var r in results)
        {
            Console.WriteLine($"Score: {r.Score}");
        }
    }
}