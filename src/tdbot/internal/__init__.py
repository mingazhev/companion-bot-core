"""Internal HTTP service layer.

Provides admin/inter-service endpoints that are NOT exposed to Telegram users:

- ``POST /internal/refine/{user_id}``  — enqueue a prompt-refinement job
- ``POST /internal/detect-change``     — classify configuration intent
"""
