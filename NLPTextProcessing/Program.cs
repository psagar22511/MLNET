using NewsCategoryClassification;
using NLPTextProcessing;

class Program
{
    static void Main(string[] args)
    {
        //Natural Language Processing (NLP Text Processing) Example
        /* What You Learn Here:
        Spam detection
        Sentiment analysis
        Fake news detection
        Resume classifier
        Intent detection chatbot */

        string dataPath = @"C:\Sagar-PC\Learning\MLNET\NLPTextProcessing\NLPTextProcessing.csv";
        string modelPath = @"C:\Sagar-PC\Learning\MLNET\NLPTextProcessing\NLPTextProcessingModel.zip";

        var nlpService = new NLPTextProcessingService(dataPath, modelPath);

        var prediction = nlpService.Predict("This is the best purchase I’ve made!");

        Console.WriteLine($"Prediction: {(prediction.PredictedLabel ? "Positive" : "Negative")}");
        Console.WriteLine($"Probability: {prediction.Probability}");
    }
}