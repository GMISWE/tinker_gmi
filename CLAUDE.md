- When a future is not ready, the /api/v1/retrieve_future endpoint should return:

  {
      "request_id": "<request_id>",
      "type": "try_again"
  }

  And when ready, it returns the actual result directly.