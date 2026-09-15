using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace Glacier.Tune.Data;

public sealed class ChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "";

    [JsonPropertyName("content")]
    public string Content { get; set; } = "";
}

public sealed class ChatConversation
{
    [JsonPropertyName("messages")]
    public List<ChatMessage> Messages { get; set; } = [];
}
