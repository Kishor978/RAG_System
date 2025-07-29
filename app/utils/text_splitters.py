from typing import List
import re

def fixed_size_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits text into fixed-size chunks with a specified overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(text) and chunks and len(chunks[-1]) < chunk_size: # Handle last small chunk
             # If the last chunk is too small, combine it with the previous one, if there is a previous one.
            if len(chunks) > 1 and len(chunks[-1]) < overlap: # Adjust threshold as needed
                chunks[-2] = chunks[-2] + chunks.pop()
                if len(chunks[-1]) > chunk_size + overlap: # Prevent overly large chunks after merge
                    chunks[-1] = chunks[-1][:chunk_size + overlap] # Truncate if too long
            
    return chunks

def recursive_character_chunking(text: str, chunk_size: int, overlap: int, separators: List[str] = None) -> List[str]:
    """
    Splits text recursively by a list of separators to maintain semantic coherence.
    Prioritizes larger separators first (e.g., paragraphs, then sentences, then words).
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""] # Paragraphs, lines, words, characters

    chunks = []
    current_text = text

    for separator in separators:
        if not current_text:
            break

        parts = current_text.split(separator)
        temp_chunks = []
        for part in parts:
            if not part.strip():
                continue

            # If a part is too large, try to split it further with the next separator
            if len(part) > chunk_size:
                if separator == separators[-1]: # If it's the smallest separator (character-level)
                    # Fallback to fixed-size chunking for very long "words" or when no suitable sep is found
                    temp_chunks.extend(fixed_size_chunking(part, chunk_size, overlap))
                else:
                    # Recursively call with the next separator
                    next_separator_index = separators.index(separator) + 1
                    temp_chunks.extend(recursive_character_chunking(part, chunk_size, overlap, separators[next_separator_index:]))
            else:
                # Add the part directly if it's within limits
                temp_chunks.append(part)
        
        # Now, try to merge small chunks with overlap
        merged_chunks = []
        if temp_chunks:
            merged_chunks.append(temp_chunks[0])
            for i in range(1, len(temp_chunks)):
                # If adding the next chunk (with overlap) keeps it within chunk_size
                if len(merged_chunks[-1]) + len(temp_chunks[i]) - overlap <= chunk_size:
                    merged_chunks[-1] += separator + temp_chunks[i] # Merge
                else:
                    merged_chunks.append(temp_chunks[i]) # New chunk

        current_text = "" # Reset to build from merged chunks
        for chunk in merged_chunks:
            chunks.append(chunk)

        # Optimization: if chunks are small enough, stop splitting
        if all(len(c) <= chunk_size for c in chunks):
            break
        
        # Flatten and re-process current_text for next separator if needed
        # This part ensures that if splitting by '\n\n' didn't break things down enough,
        # we then try '\n' on the resulting larger parts.
        current_text = separator.join(chunks) if chunks else ""
        chunks = [] # Clear for next iteration's re-chunking
    
    # Final pass to ensure no chunks are too small if possible, by combining with overlap
    final_chunks = []
    if current_text: # If something remains after all separators
        final_chunks.extend(fixed_size_chunking(current_text, chunk_size, overlap))
    else: # If chunks were built incrementally
        final_chunks = chunks # Use the last set of chunks

    # Post-processing: ensure chunks are not empty and handle overlaps gracefully
    # This loop specifically handles the overlap for the last piece of text
    processed_chunks = []
    i = 0
    while i < len(final_chunks):
        current_chunk_content = final_chunks[i].strip()
        if current_chunk_content:
            processed_chunks.append(current_chunk_content)
        i += 1
    
    # Additional logic to ensure minimum chunk size / handling very small remaining chunks
    # This is a simplified approach, for production consider a dedicated text splitter library.
    # For now, let's just make sure we don't have many tiny chunks unless unavoidable.
    refined_chunks = []
    if processed_chunks:
        refined_chunks.append(processed_chunks[0])
        for i in range(1, len(processed_chunks)):
            if len(refined_chunks[-1]) + len(processed_chunks[i]) + len(separator) - overlap <= chunk_size:
                refined_chunks[-1] += separator + processed_chunks[i]
            else:
                refined_chunks.append(processed_chunks[i])
    
    return refined_chunks


# Example usage (for testing)
if __name__ == "__main__":
    text = "This is a sentence. This is another sentence.\n\nThis is a new paragraph. It has more text."
    
    print("Fixed-size chunking:")
    chunks_fixed = fixed_size_chunking(text, chunk_size=30, overlap=5)
    for i, chunk in enumerate(chunks_fixed):
        print(f"Chunk {i+1} (len {len(chunk)}): '{chunk}'")
    
    print("\nRecursive character chunking:")
    chunks_recursive = recursive_character_chunking(text, chunk_size=30, overlap=5)
    for i, chunk in enumerate(chunks_recursive):
        print(f"Chunk {i+1} (len {len(chunk)}): '{chunk}'")