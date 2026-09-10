using dotnet_api.Model;

namespace dotnet_api.Service
{
    public class MlService
    {
        private readonly HttpClient _httpClient;

        public MlService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<string> GetPrediction(PredictionRequest request)
        {
            var response = await _httpClient.PostAsJsonAsync(
                "http://127.0.0.1:8000/predict",
                request);

            var result = await response.Content.ReadAsStringAsync();

            if (!response.IsSuccessStatusCode)
            {
                throw new HttpRequestException(
                    $"FastAPI returned {(int)response.StatusCode}: {result}");
            }

            return result;
        }
    }
}
