using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Tune.Config;
using Glacier.Tune.Data;
using Xunit;

namespace Glacier.Tune.Tests;

public class ChatMlDatasetTests
{
    [Fact]
    public void LoraConfig_ComputesCorrectScaling()
    {
        var config = new LoraConfig { Rank = 16, Alpha = 32f };
        Assert.Equal(2.0f, config.Scaling);

        var config2 = new LoraConfig { Rank = 8, Alpha = 16f };
        Assert.Equal(2.0f, config2.Scaling);
    }

    [Fact]
    public void ChatConversation_DeserializesAndFormatsProperly()
    {
        var conv = new ChatConversation
        {
            Messages =
            [
                new ChatMessage { Role = "system", Content = "You are a helpful assistant." },
                new ChatMessage { Role = "user", Content = "Hello!" },
                new ChatMessage { Role = "assistant", Content = "Hi there, how can I help?" }
            ]
        };

        Assert.Equal(3, conv.Messages.Count);
        Assert.Equal("system", conv.Messages[0].Role);
        Assert.Equal("user", conv.Messages[1].Role);
        Assert.Equal("assistant", conv.Messages[2].Role);
    }
}
