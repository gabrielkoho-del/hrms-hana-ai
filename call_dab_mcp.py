import asyncio
import json
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

async def main():
    try:
        async with streamablehttp_client("http://localhost:5002/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Query V_EMP selecting all columns, no pagination
                print("=== read_records V_EMP: all columns, first=101 ===")
                result = await session.call_tool(
                    "read_records",
                    arguments={
                        "entity": "V_EMP",
                        "first": 101
                    }
                )
                
                resp = None
                for content in result.content:
                    if hasattr(content, 'text'):
                        resp = json.loads(content.text)
                        break
                
                if not resp:
                    print("No valid response")
                    return
                
                data = resp.get('result', {})
                items = data.get('value', [])
                
                print(f"Records returned: {len(items)}")
                if items:
                    print(f"Columns in first record: {len(items[0].keys())}")
                    print(f"Sample keys: {list(items[0].keys())[:10]}")
                    print(f"\nFirst record:")
                    for k, v in list(items[0].items())[:15]:
                        print(f"  {k}: {v}")
                    print(f"\nLast record:")
                    for k, v in list(items[-1].items())[:15]:
                        print(f"  {k}: {v}")
                        
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
