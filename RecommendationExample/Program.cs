using RecommendationExample;

class Program
{
    static void Main(string[] args)
    {
        //Recommendation Example

        string dataPath = @"C:\Sagar-PC\Learning\MLNET\RecommendationExample\spamdata.csv";
        string modelPath = @"C:\Sagar-PC\Learning\MLNET\RecommendationExample\recommendationExample.zip";

        var service = new RecommendationService(dataPath, modelPath);

        var result = service.Predict(new ProductRating { userId = 1, productId = 3 });

        Console.WriteLine($"Predicted Rating: {result.Score}");
    }
}