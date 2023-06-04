﻿using System.Net.Http;
using System.Text;
using System.Text.Json;

namespace IO.Milvus.Client.REST;

internal static class HttpRequest
{
    public static HttpRequestMessage CreateGetRequest(string url, object payload = null)
    {
        return new HttpRequestMessage(HttpMethod.Get, url)
        {
            Content = GetJsonContent(payload)
        };
    }

    public static HttpRequestMessage CreatePostRequest(string url, object payload = null)
    {
        return new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = GetJsonContent(payload)
        };
    }

    public static HttpRequestMessage CreatePutRequest(string url, object payload = null)
    {
        return new HttpRequestMessage(HttpMethod.Put, url)
        {
            Content = GetJsonContent(payload)
        };
    }

    public static HttpRequestMessage CreatePatchRequest(string url, object payload = null)
    {
        return new HttpRequestMessage(new HttpMethod("PATCH"), url)
        {
            Content = GetJsonContent(payload)
        };
    }

    public static HttpRequestMessage CreateDeleteRequest(string url,object payload = null)
    {
        return new HttpRequestMessage(HttpMethod.Delete, url) 
        { 
            Content = GetJsonContent(payload)
        };
    }

    private static StringContent GetJsonContent(object payload)
    {
        if (payload == null)
        {
            return null;
        }
        
        string strPayload = payload is string s ? s : JsonSerializer.Serialize(payload);
        return new StringContent(strPayload, Encoding.UTF8, "application/json");
    }
}