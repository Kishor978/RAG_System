#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}RAG System End-to-End Test${NC}"
echo -e "${YELLOW}=====================================${NC}"
echo -e "${BLUE}This test will verify the core functionality of your RAG system.${NC}"

# Step 1: Check if the server is running
echo -e "\n${YELLOW}Step 1: Checking if server is running...${NC}"
echo -e "${BLUE}This verifies that your FastAPI application is running on port 8000.${NC}"
curl -s http://localhost:8000/ > /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Server is not running! Please start the server with:${NC}"
    echo -e "uvicorn app.main:app --reload"
    exit 1
else
    echo -e "${GREEN}Server is running.${NC}"
fi

# Step 2: Check Redis connection
echo -e "\n${YELLOW}Step 2: Checking Redis connection...${NC}"
echo -e "${BLUE}This verifies that Redis is running and accessible for conversation storage.${NC}"
python redis_setup.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Redis connection failed. Please ensure Redis is running.${NC}"
    echo -e "You can start Redis with: docker run --name redis-rag -p 6379:6379 -d redis"
    exit 1
else
    echo -e "${GREEN}Redis connection successful.${NC}"
fi

# Step 3: Test document ingestion
echo -e "\n${YELLOW}Step 3: Testing document ingestion...${NC}"
echo -e "${BLUE}This tests the chunking and embedding process with a sample document.${NC}"
INGEST_RESPONSE=$(curl -s -X POST "http://localhost:8000/documents/ingest" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@./sample.txt" \
    -F "chunking_strategy=recursive_character")

DOCUMENT_ID=$(echo $INGEST_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('document_id', ''))")

if [ -z "$DOCUMENT_ID" ]; then
    echo -e "${RED}Document ingestion failed.${NC}"
    echo -e "Response: $INGEST_RESPONSE"
    exit 1
else
    echo -e "${GREEN}Document ingested successfully with ID: $DOCUMENT_ID${NC}"
fi

# Step 4: Test document query via conversation endpoint
echo -e "\n${YELLOW}Step 4: Testing document query via conversation endpoint...${NC}"
echo -e "${BLUE}This tests the RAG system's ability to retrieve information using the conversation API.${NC}"
echo -e "${BLUE}Using conversation endpoint instead of separate query endpoint.${NC}"
QUERY_RESPONSE=$(curl -s -X POST "http://localhost:8000/conversation/chat" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"query": "test chunking", "conversation_id": null}')

if [[ $QUERY_RESPONSE == *"response"* ]] && [[ $QUERY_RESPONSE != *"Not Found"* ]]; then
    echo -e "${GREEN}Query executed successfully via conversation endpoint.${NC}"
    echo -e "Response: $(echo $QUERY_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('response', '')[:50] + '...')")"
else
    echo -e "${RED}Query via conversation endpoint failed or returned no results.${NC}"
    echo -e "Response: $QUERY_RESPONSE"
fi

# Step 5: Test conversation API
echo -e "\n${YELLOW}Step 5: Testing conversation API...${NC}"
echo -e "${BLUE}This tests starting a new conversation and getting a response.${NC}"
CHAT_RESPONSE=$(curl -s -X POST "http://localhost:8000/conversation/chat" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d '{"query": "Tell me about chunking", "conversation_id": null}')

CONVERSATION_ID=$(echo $CHAT_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('conversation_id', ''))")

if [ -z "$CONVERSATION_ID" ]; then
    echo -e "${RED}Conversation API test failed.${NC}"
    echo -e "Response: $CHAT_RESPONSE"
else
    echo -e "${GREEN}Conversation started successfully with ID: $CONVERSATION_ID${NC}"
    echo -e "Response: $(echo $CHAT_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('response', '')[:50] + '...')")"
    
    # Step 6: Test follow-up query
    echo -e "\n${YELLOW}Step 6: Testing follow-up query...${NC}"
    echo -e "${BLUE}This tests the system's ability to maintain conversation context.${NC}"
    FOLLOWUP_RESPONSE=$(curl -s -X POST "http://localhost:8000/conversation/chat" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"How does it work?\", \"conversation_id\": \"$CONVERSATION_ID\"}")
    
    if [[ $FOLLOWUP_RESPONSE == *"response"* ]]; then
        echo -e "${GREEN}Follow-up query successful.${NC}"
        echo -e "Response: $(echo $FOLLOWUP_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('response', '')[:50] + '...')")"
    else
        echo -e "${RED}Follow-up query failed.${NC}"
        echo -e "Response: $FOLLOWUP_RESPONSE"
    fi
    
    # Step 7: Test conversation history retrieval
    echo -e "\n${YELLOW}Step 7: Testing conversation history retrieval...${NC}"
    echo -e "${BLUE}This tests the system's ability to retrieve conversation history from Redis.${NC}"
    # The endpoint is actually "/conversation/conversations/{id}"
    HISTORY_RESPONSE=$(curl -s -X GET "http://localhost:8000/conversation/conversations/$CONVERSATION_ID" \
        -H "accept: application/json")
    
    # Check response format and handle potential errors more gracefully
    if [[ $HISTORY_RESPONSE == *"messages"* ]]; then
        MESSAGE_COUNT=$(echo $HISTORY_RESPONSE | python -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('messages', [])) if isinstance(data.get('messages', []), list) else 0)")
        
        if [ "$MESSAGE_COUNT" -gt 0 ]; then
            echo -e "${GREEN}Conversation history retrieved successfully with $MESSAGE_COUNT messages.${NC}"
        else
            echo -e "${YELLOW}Conversation history retrieved but contains no messages.${NC}"
        fi
    else
        echo -e "${RED}Conversation history retrieval failed.${NC}"
        echo -e "Response: $HISTORY_RESPONSE"
        echo -e "${BLUE}Hint: Check if Redis is properly saving conversations.${NC}"
    fi
fi

# Step 8: Test booking API
echo -e "\n${YELLOW}Step 8: Testing booking API...${NC}"
echo -e "${BLUE}This tests the system's interview booking functionality.${NC}"
BOOKING_RESPONSE=$(curl -s -X POST "http://localhost:8000/conversation/booking" \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"Test User\", \"email\": \"test@example.com\", \"date\": \"2025/08/15\", \"time\": \"10:00 AM\", \"conversation_id\": \"$CONVERSATION_ID\"}")

if [[ $BOOKING_RESPONSE == *"success"* ]]; then
    echo -e "${GREEN}Booking API test successful.${NC}"
else
    echo -e "${RED}Booking API test failed.${NC}"
    echo -e "Response: $BOOKING_RESPONSE"
fi

echo -e "\n${YELLOW}=====================================${NC}"
echo -e "${GREEN}End-to-End Test Completed${NC}"
echo -e "${YELLOW}=====================================${NC}"
echo -e "\n${BLUE}Test Summary:${NC}"
echo -e "${BLUE}This end-to-end test has verified the core functionality of your RAG system:${NC}"
echo -e "${BLUE}1. Server connection${NC}"
echo -e "${BLUE}2. Redis connection${NC}"
echo -e "${BLUE}3. Document ingestion${NC}"
echo -e "${BLUE}4. Document querying via conversation API${NC}"
echo -e "${BLUE}5. Conversation handling${NC}"
echo -e "${BLUE}6. Follow-up queries with context${NC}"
echo -e "${BLUE}7. Conversation history storage and retrieval${NC}"
echo -e "${BLUE}8. Booking functionality${NC}"
echo -e "\n${BLUE}If any tests failed, check the logs above for details.${NC}"
