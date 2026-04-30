using System.Text;
using System.Text.Json;

namespace AiChatbot.Services
{
    public class OpenAiService
    {
        private readonly HttpClient _httpClient;
        private readonly string _apiKey;

        public OpenAiService(IConfiguration config)
        {
            _httpClient = new HttpClient();
            _apiKey = config["OpenAI:ApiKey"];
        }

        public async Task<string> GetResponse(string userMessage)
        {
            var httpClient = new HttpClient();

            var responsee = await httpClient.PostAsync(
                "http://localhost:11434/api/generate", //Default port where "Ollama" installed.
                new StringContent(
                    JsonSerializer.Serialize(new
                    {
                        model = "llama3",
                        prompt = userMessage,
                        stream = false
                    }),
                    Encoding.UTF8,
                    "application/json"
                )
            );

            var result = await responsee.Content.ReadAsStringAsync();
            Console.WriteLine(result);
            using var doc = JsonDocument.Parse(result);

            return doc.RootElement
                .GetProperty("response")
                .GetString();
        }
    }
}
