using ConsoleApp.Services;

class Program
{
    static void Main(string[] args)
    {
        //Binary Classification Example (Email spam detection) Spam detection,Sentiment analysis)

        //string dataPath = @"C:\Sagar-PC\Learning\MLNET\ConsoleApp\spamdata.csv";
        //string modelPath = @"C:\Sagar-PC\Learning\MLNET\ConsoleApp\spamDetectionModel.zip";

        string dataPath = @"C:\SagarPatel\Practice\ML.NET\ConsoleApp\spamdata.csv";
        string modelPath = @"C:\SagarPatel\Practice\ML.NET\ConsoleApp\spamDetectionModel.zip";

        var spamDetector = new SpamDetectionService(dataPath, modelPath);

        var result = spamDetector.Predict("Win money now"); //It's spam. Match in csv file.
        //"How are you?" = For not spam.

        Console.WriteLine($"Prediction: {(result.PredictedLabel ? "Spam" : "Not Spam")}");
        Console.WriteLine($"Probability: {result.Probability:P2}");
    }
}