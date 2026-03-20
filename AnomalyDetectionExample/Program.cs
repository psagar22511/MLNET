using AnomalyDetectionExample;

class Program
{
    static void Main(string[] args)
    {
        //Anomaly detection Example

        string dataPath = @"C:\Sagar-PC\Learning\MLNET\AnomalyDetectionExample\NLPTextProcessing.csv";
        string modelPath = @"C:\Sagar-PC\Learning\MLNET\AnomalyDetectionExample\AnomalyDetectionModel.zip";

        var nlpService = new AnomalyDetectionService(dataPath, modelPath);
    }
}