using Connectors.Memory.Milvus;
using Connectors.Memory.MilvusTests;
using IO.Milvus.Client;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Memory;
using Xunit;

namespace Connectors.Memory.Milvus.Tests;

public partial class MilvusMemoryStoreTests
{
    [Theory]
    [ClassData(typeof(TestClients))]
    public async Task SemanticKernelTest(IMilvusClient milvusClient)
    {
        string apiKey = "OpenAI apikey";

        var db = new MilvusMemoryStore(milvusClient, 1536, milvusClient.ToString().Contains("zilliz", StringComparison.OrdinalIgnoreCase));

        var builder = new KernelBuilder();
        IKernel kernel = builder

            .WithOpenAITextCompletionService("text-davinci-003", apiKey)
            .WithOpenAITextEmbeddingGenerationService("text-embedding-ada-002", apiKey)
            .WithMemoryStorage(db)
            .Build();

        const string memoryCollectionName = "FactsAboutMe";//Milvus collection name can only contain numbers, letters and underscores
        await kernel.Memory.SaveInformationAsync(memoryCollectionName, id: "LinkedIn Bio",
            text: "I currently work in the hotel industry at the front desk. I won the best team player award.");
        await kernel.Memory.SaveInformationAsync(memoryCollectionName, id: "LinkedIn History",
            text: "I have worked as a tourist operator for 8 years. I have also worked as a banking associate for 3 years.");
        await kernel.Memory.SaveInformationAsync(memoryCollectionName, id: "Recent Facebook Post",
            text: "My new dog Trixie is the cutest thing you've ever seen. She's just 2 years old.");
        await kernel.Memory.SaveInformationAsync(memoryCollectionName, id: "Old Facebook Post",
            text: "Can you believe the size of the trees in Yellowstone? They're huge! I'm so committed to forestry concerns.");
        Console.WriteLine("Four GIGANTIC vectors were generated just now from those 4 pieces of text above.");

        var myFunction = kernel.CreateSemanticFunction(@"
            Tell me about me and {{$input}} in less than 70 characters.",
            maxTokens: 100, 
            temperature: 0.8, 
            topP: 1);
        var result = await myFunction.InvokeAsync("my work history");
        Console.WriteLine(result);

        string ask = "Tell me about me and my work history.";
        var relatedMemory = "I know nothing.";
        var counter = 0;

        var memories = kernel.Memory.SearchAsync(memoryCollectionName, ask, limit: 5, minRelevanceScore: 0.77);

        await foreach (MemoryQueryResult memory in memories)
        {
            if (counter == 0) { relatedMemory = memory.Metadata.Text; }
            Console.WriteLine($"Result {++counter}:\n  >> {memory.Metadata.Id}\n  Text: {memory.Metadata.Text}  Relevance: {memory.Relevance}\n");
        }

        myFunction = kernel.CreateSemanticFunction(
            @"""
            {{$input}}
            Tell me about me and my work history in less than 70 characters""",
            maxTokens: 100, 
            temperature: 0.1, topP: .1);

        result = await myFunction.InvokeAsync(relatedMemory);

        Console.WriteLine(result);
    }
}
