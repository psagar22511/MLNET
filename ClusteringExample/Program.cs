
using ClusteringExample;

class Program
{
    static void Main(string[] args)
    {
        //Clustering Example

        string dataPath = @"C:\Sagar-PC\Learning\MLNET\ClusteringExample\spamdata.csv";
        string modelPath = @"C:\Sagar-PC\Learning\MLNET\ClusteringExample\clusteringExampleModel.zip";

        var service = new ClusteringService(dataPath, modelPath);

        var newCustomer = new CustomerData { Age = 30, AnnualIncome = 60000 };
        var result = service.Predict(newCustomer);

        Console.WriteLine($"Predicted Cluster: {result.PredictedClusterId}");
        Console.WriteLine("Distances to centroids: " + string.Join(", ", result.Distances));
    }
}