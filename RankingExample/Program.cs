using RankingExample;

class Program
{
    static void Main(string[] args)
    {
        //Ranking Example

        string dataPath = @"C:\Sagar-PC\Learning\MLNET\RankingExample\spamdata.csv";
        string modelPath = @"C:\Sagar-PC\Learning\MLNET\RankingExample\rankingModel.zip";

        var spamDetector = new RankingService(dataPath, modelPath);

        var result = spamDetector.Predict(new RankingData
        {
            Feature1 = 2,
            Feature2 = 0.15f,
            GroupId = 1
        });

        Console.WriteLine($"Ranking Score: {result.Score}");
    }
}