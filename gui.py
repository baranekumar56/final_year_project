import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

# Create main window
root = tk.Tk()
root.title("Query Interface")
root.geometry("800x600")
root.configure(bg="#2b2b2b")

# Custom fonts
title_font = tkFont.Font(family="Segoe UI", size=16, weight="bold")
label_font = tkFont.Font(family="Segoe UI", size=12)
button_font = tkFont.Font(family="Segoe UI", size=11)

# Main container with sidebar and content
main_frame = tk.Frame(root, bg="#2b2b2b")
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Left sidebar (unchanged)
sidebar = tk.Frame(main_frame, width=200, bg="#1e1e1e", relief=tk.RAISED, bd=1)
sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
sidebar.pack_propagate(False)

tk.Label(sidebar, text="Query Interface", font=title_font, bg="#1e1e1e", fg="#ffffff").pack(pady=20)
tk.Label(sidebar, text="• Modern Design", font=label_font, bg="#1e1e1e", fg="#cccccc").pack(anchor="w", padx=20, pady=5)
tk.Label(sidebar, text="• Clean Layout", font=label_font, bg="#1e1e1e", fg="#cccccc").pack(anchor="w", padx=20, pady=2)
tk.Label(sidebar, text="• Responsive", font=label_font, bg="#1e1e1e", fg="#cccccc").pack(anchor="w", padx=20, pady=2)

# Right content area
content_frame = tk.Frame(main_frame, bg="#2b2b2b")
content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Input area (unchanged)
input_frame = tk.Frame(content_frame, bg="#363636", relief=tk.RAISED, bd=1)
input_frame.pack(fill=tk.X, pady=(0, 10))

title_label = tk.Label(input_frame, text="Enter your query:", font=title_font, 
                      bg="#363636", fg="#ffffff", anchor="w")
title_label.pack(anchor="w", padx=20, pady=(20, 10))

text_entry = tk.Text(input_frame, height=4, font=("Segoe UI", 11), bg="#1e1e1e", fg="#ffffff", 
                    insertbackground="#ffffff", relief=tk.FLAT, bd=1, wrap=tk.WORD)
text_entry.pack(fill=tk.X, padx=20, pady=(0, 20))

submit_btn = tk.Button(input_frame, text="Submit Query", font=button_font, bg="#007acc", fg="white",
                      activebackground="#005a9e", relief=tk.FLAT, bd=0, height=2,
                      command=lambda: show_result())
submit_btn.pack(anchor="w", padx=20, pady=(0, 20))

# Results notebook
notebook = ttk.Notebook(content_frame)
notebook.pack(fill=tk.BOTH, expand=True)

# Results page
result_frame = tk.Frame(notebook, bg="#2b2b2b")
notebook.add(result_frame, text="Results")

# Result display area
result_text = tk.Text(result_frame, font=("Segoe UI", 11), bg="#1e1e1e", fg="#ffffff", 
                     insertbackground="#ffffff", relief=tk.FLAT, bd=1, wrap=tk.WORD,
                     state=tk.DISABLED)
result_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

def write_to_result(data, clear_first=False):
    """Write data to the result text widget"""
    result_text.config(state=tk.NORMAL)
    if clear_first:
        result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, data)
    result_text.config(state=tk.DISABLED)
    result_text.see(tk.END)  # Auto-scroll to bottom

def show_result():
    query = text_entry.get("1.0", tk.END).strip()
    
    # Clear and write query
    write_to_result(f"User Query:\n{query}\n\n", clear_first=True)
    
    # Write sample results (replace this with your actual processing)
    write_to_result("Processing query...\n\n")
    write_to_result("Results:\n")
    write_to_result("- Found 42 matches\n")
    write_to_result("- Top result: Example data\n")
    write_to_result("- Processing time: 0.2s\n")
    

# Example: Write something on startup
write_to_result("Ready to receive queries...\n")





