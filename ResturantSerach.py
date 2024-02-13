import os
from tkinter import *
import pymongo
from collections import defaultdict
import math
import re

class RestaurantSearch:
    def __init__(self):
        # Initializes the RestaurantSearch class, creating the inverted index and document list
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.db = self.client['restaurants_db']
        self.collection = self.db['restaurants']
        self.inverted_index = dict()
        self.documents = []              
        self.doc_lengths = dict()
        self.total_doc_len = 0.0
        self.load_and_index_data()
        self.document_count = self.collection.count_documents({})
        self.avg_doc_len = self.total_doc_len / self.document_count

    def load_and_index_data(self):
        # Loads restaurant data from MongoDB and indexes them for search
        restaurants = self.collection.find({})
        for restaurant in restaurants:
            content = restaurant['name'] + " " + restaurant['location']
            self.documents.append(content)
            self.index_document(content)

    def index_document(self, content):
        # Creates an index of words from the document content for search functionality
        tokens = content.lower().split()
        doc_id = len(self.documents) - 1
        self.doc_lengths[doc_id] = len(tokens)
        # redo with form [token][doc_id]
        self.inverted_index[doc_id] = dict()
        for token in tokens:
            cleaned_token = token.strip(".,").replace("and", "").replace("the", "")
            if cleaned_token not in self.inverted_index:
                self.inverted_index[cleaned_token] = dict()
            if doc_id in self.inverted_index[cleaned_token]:
                self.inverted_index[cleaned_token][doc_id] += 1
            else:
                self.inverted_index[cleaned_token][doc_id] = 1
        self.total_doc_len += len(tokens)

    def search(self, query, location):
        # Searches indexed documents based on the query and location, returning relevant results
        query_terms = query.lower().split()
        query_counts = dict()
        qt_df = dict()
        query_len = len(query_terms)      

        result_list = []
        for term in query_terms:
            cleaned_term = term.strip(".,").replace("and", "").replace("the", "")
            if cleaned_term in self.inverted_index:
                query_counts[cleaned_term] = query_counts.get(cleaned_term, 0) + 1
                qt_df[cleaned_term] = len(self.inverted_index[cleaned_term])
                result_list.extend(self.inverted_index[cleaned_term])

        unique_results = set(result_list)
        num_unique_results = len(unique_results)

        rank_dict = {doc_id: self.bm25(doc_id, query_counts, num_unique_results, qt_df) for doc_id in unique_results}
        ranked_results = sorted(rank_dict, key=rank_dict.get, reverse=True)

        return self.format_results(ranked_results, rank_dict, location), len(ranked_results)

    def format_results(self, ranked_results, rank_dict, location):
        # Formats the search results for display, filtering by location if specified
        output = []
        location_match_count = 0
        # Scale every result as a percentage of the most relevant result
        top_score = 0

        for doc_id in ranked_results:
            title, loc = self.documents[doc_id].split("\n")[:2]
            if location and not self.match_location(loc, location):
                continue
            top_score = max(top_score, rank_dict[doc_id])

        for doc_id in ranked_results:
            title, loc = self.documents[doc_id].split("\n")[:2]
            if location and not self.match_location(loc, location):
                continue
            location_match_count += 1
            relevance = '{:.3f}'.format(round((rank_dict[doc_id] / top_score) * 100, 3))
            output.append(f"{title}, LOC: {loc}, relevance: {relevance}%")
        return output, location_match_count

    def bm25(self, doc_id, query_counts, num, qt_df):
        b_ = .75
        k1_ = 1.6
        k3_ = 1

        okapi_bm25 = 0

        for t in query_counts:
            if doc_id in self.inverted_index[t]:
                # Inverse Document Frequency Term
                IDF = math.log(
                    1.0 + (self.document_count - qt_df[t] + 0.5) / (qt_df[t] + 0.5));

                # Term Frequency with Document Length Normalization
                TF = ((k1_ + 1.0) * self.inverted_index[t][doc_id]) / ((k1_ * ((1.0 - b_) + b_ * self.doc_lengths[doc_id] / self.avg_doc_len))
                              + self.inverted_index[t][doc_id]);

                # QTF handles how to value appearances of a term multiple times in the same query
                QTF = ((k3_ + 1.0) * query_counts[t]) / (k3_ + query_counts[t]);

                #Okapi BM25
                okapi_bm25 += TF * IDF * QTF

        return okapi_bm25

    def match_location(self, a, b):
        regex = re.compile('[^a-zA-Z]')
        a_new = regex.sub('', a).lower()
        b_new = regex.sub('', b).lower()
        return a_new == b_new


class GUI:
    def __init__(self, master):
        # Initializes the GUI elements and sets up the main window
        self.master = master
        master.title('The Restaurant Selector')
        master.geometry("850x600")

        self.search_engine = RestaurantSearch()

        Label(master, text="What do you feel like eating today?", font=("Times", 21)).pack()
        self.query_entry = Text(master, width=40, height=5)
        self.query_entry.pack()

        Label(master, text="(Optional) Please enter a city name and the first two letters of the state. e.g., Urbana IL", font=("Times", 8)).pack()
        self.location_entry = Entry(master, width=20)
        self.location_entry.pack()

        self.results_text = Text(master, width=95, height=24)
        self.results_text.pack()

        Button(master, text="Find Restaurants", command=self.perform_search).pack()

    def perform_search(self):
        # Initiates a search when the 'Find Restaurants' button is clicked
        query = self.query_entry.get("1.0", "end").strip()
        location = self.location_entry.get().strip()
        results, total_results = self.search_engine.search(query, location)
        results, location_match_count = results

        self.results_text.delete("1.0", END)
        self.results_text.insert("1.0", f"{total_results} total results returned\n")
        if location:
            self.results_text.insert(END, f"Location Filter Applied\n{location_match_count} relevant to your specified location\n")
        else:
            self.results_text.insert(END, "No Location Filter Applied\n")
        for result in results:
            self.results_text.insert(END, result + "\n")

def main():
    # The main function to run the application
    root = Tk()
    gui = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
