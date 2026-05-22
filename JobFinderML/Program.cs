using JobFinderML;

class Program
{
    static void Main(string[] args)
    {
        string dataPath = @"C:\Sagar-PC\Learning\MLNET\JobFinderML\jobdata.csv";
        string modelPath = @"C:\Sagar-PC\Learning\MLNET\JobFinderML\jobFinderModel.zip";

        var jobFinder = new JobFinderService(dataPath, modelPath);
        var result = jobFinder.Predict(".NET Backend Developer", "C# ASP.NET Core Web API Azure");
        Console.WriteLine($"Relevant: {result.Prediction}");
    }
}