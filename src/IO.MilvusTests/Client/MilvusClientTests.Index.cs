﻿using IO.Milvus.ApiSchema;
using IO.Milvus;
using IO.Milvus.Client;
using Xunit;

namespace IO.MilvusTests.Client;

public partial class MilvusClientTests
{
    [Theory]
    [ClassData(typeof(TestClients))]
    public async Task IndexTest(IMilvusClient2 milvusClient)
    {
        string collectionName = milvusClient.GetType().Name;

        bool collectionExist = await milvusClient.HasCollectionAsync(collectionName);

        if (collectionExist)
        {
            await milvusClient.DropCollectionAsync(collectionName);
        }

        Random ran = new Random();
        List<long> book_id_array = new();
        List<long> word_count_array = new();
        List<List<float>> book_intro_array = new();
        for (long i = 0L; i < 2000; ++i)
        {
            book_id_array.Add(i);
            word_count_array.Add(i + 10000);
            List<float> vector = new();
            for (int k = 0; k < 2; ++k)
            {
                vector.Add(ran.Next());
            }
            book_intro_array.Add(vector);
        }

        await milvusClient.CreateCollectionAsync(
            collectionName,
            new[] {
                new FieldType("book_id",MilvusDataType.Int64,true),
                new FieldType("word_count",MilvusDataType.Int64,false),
                FieldType.CreateFloatVector("book_intro",2),
            }
        );

        bool exist = await milvusClient.HasCollectionAsync(collectionName);
        Assert.True(exist, "Create collection failed");

        MilvusMutationResult result = await milvusClient.InsertAsync(collectionName,
            new[]
            {
                Field.Create("book_id",book_id_array),
                Field.Create("word_count",word_count_array),
                Field.CreateFloatVector("book_intro",book_intro_array),
            });

        Assert.True(result.InsertCount == 2000, "Insert data failed");
        Assert.True(result.SuccessIndex.Count == 2000, "Insert data failed");

        await milvusClient.CreateIndexAsync(collectionName, "book_id","default" ,MilvusIndexType.IVF_FLAT, MilvusMetricType.L2, new Dictionary<string, string> { { "nlist", "1024" } });

        var indexs = await milvusClient.DescribeIndexAsync(collectionName, "book_id");

        Assert.Equal(1,indexs.Count);

        await milvusClient.DropCollectionAsync(collectionName);
    }
}