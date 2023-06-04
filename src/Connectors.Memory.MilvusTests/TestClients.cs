using IO.Milvus.Client;
using Xunit;

namespace Connectors.Memory.MilvusTests;

internal class TestClients : TheoryData<IMilvusClient>
{
    public TestClients()
    {
        IEnumerable<MilvusConfig> configs = MilvusConfig.Load();

        foreach (var item in configs)
        {
            Add(item.CreateClient());
        }
    }
}