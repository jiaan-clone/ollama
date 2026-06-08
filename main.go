package main

import (
	"context"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cmd"
)

func main() {
	/*
		这里 ExecuteContext() 会让 Cobra 解析命令行参数。比如：ollama run gemma3 "hi"，
		Cobra 会找到 runCmd，然后按顺序执行：
			Args 校验
			  	-> PreRunE: checkServerHeartbeat
			  	-> RunE: RunHandler
	*/
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}
