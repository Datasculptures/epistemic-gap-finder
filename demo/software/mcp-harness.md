MCP Test Harness is a Python CLI tool that performs automated security and
conformance testing of Model Context Protocol (MCP) STDIO servers, validating
their behaviour against the MCP specification and a suite of adversarial
security probes without requiring a conformant client.

It takes an MCP server executable as input, communicates with it directly
over raw STDIO using JSON-RPC 2.0, and exercises it across more than sixty
test cases spanning protocol conformance, error handling, and security
boundaries.

It produces a structured test report showing pass, fail, and warning status
per test case, organised by conformance and security suites, with human-readable
failure descriptions and an overall verdict.

The MCP Test Harness does not test HTTP-based MCP servers, does not generate
MCP server implementations, and does not measure the semantic quality or
usefulness of the tools a server exposes — only their protocol compliance and
security posture.
