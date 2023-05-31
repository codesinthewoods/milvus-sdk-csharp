﻿using IO.Milvus;
using IO.Milvus.Client;
using Xunit;

namespace IO.MilvusTests.Client;

public partial class MilvusClientTests
{
    [Theory]
    [ClassData(typeof(TestClients))]
    public async Task PartitionTest(IMilvusClient milvusClient)
    {
        string collectionName = milvusClient.GetType().Name;
        var partition = $"{collectionName}Partition";

        //Check if collection exists.
        bool collectionExist = await milvusClient.HasCollectionAsync(collectionName);
        if (!collectionExist)
        {
            await milvusClient.CreateCollectionAsync(
                collectionName,
                new[] {
                new FieldType("book",MilvusDataType.Int64,true) }
                );
        }

        //Check if this partition exists.
        bool partitionExist = await milvusClient.HasPartitionAsync(collectionName, partition);
        if (partitionExist)
        {
            await milvusClient.ReleaseCollectionAsync(collectionName);
            await milvusClient.DropPartitionsAsync(collectionName,partition);
        }

        //Create partition
        if (milvusClient?.ToString()?.Contains("zilliz") == true)
        {
            return;
        }
        await milvusClient.CreatePartitionAsync(collectionName, partition);
        partitionExist = await milvusClient.HasPartitionAsync(collectionName, partition);
        Assert.True(partitionExist, $"Failed Create Collection: {collectionName}, Partition: {partition}");

        //Drop
        await milvusClient.DropPartitionsAsync(collectionName, partition);
        partitionExist = await milvusClient.HasPartitionAsync(collectionName, partition);
        Assert.False(partitionExist, $"Failed Drop Collection: {collectionName}, Partition: {partition}");

        // Cooldown, sometimes the DB doesn't refresh completely
        await Task.Delay(1000);
    }
}