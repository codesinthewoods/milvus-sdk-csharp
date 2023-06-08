﻿namespace IO.Milvus;

/// <summary>
/// Compaction state.
/// </summary>
public enum MilvusCompactionState
{
    /// <summary>
    /// Unknown.
    /// </summary>
    UndefiedState = 0,

    /// <summary>
    /// Executing.
    /// </summary>
    Executing = 1,

    /// <summary>
    /// Completed.
    /// </summary>
    Completed = 2,
}
