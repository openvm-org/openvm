import argparse

from main import read_aggregations;

def get_summary_rows(md_files, headers, gh_pages_link):
    outputs = []

    for md in md_files:
        with open(md, 'r') as file:
            lines = iter(file.readlines())
        row = []
        for line in lines:
            cols = [col.strip() for col in line.split('|') if col.strip()]
            print(cols)
            if cols == headers:
                next(lines)

                line = next(lines).strip()
                app_cols = [col.strip() for col in line.split('|') if col.strip()]
                link = f"[ {app_cols[0]} ]({gh_pages_link}/individual/{md})"
                row = [link] + app_cols[1:]

                line = next(lines).strip()
                agg_cols = [col.strip() for col in line.split('|') if col.strip()]
                if len(agg_cols) == len(app_cols) and "leaf" in agg_cols[0]:
                    row += agg_cols[1:]
                else:
                    row += (["-"] * (len(app_cols) - 1))
                break
        
        for line in lines:
            if "Instance Type:" in line:
                row += [f"{line.split(':')[1].strip()}"]
                continue

            if "Memory Allocator:" in line:
                row += [f"{line.split(':')[1].strip()}"]
                break
        
        if len(row) != 0:
            outputs.append(row)
    print(outputs)
    return outputs

def write_to_md_table(rows, headers, file_path='summary.md'):
    with open(file_path, 'w') as f:
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '---|' * len(headers) + '\n')
        for row in rows:
            f.write('| ' + ' | '.join(row) + ' |\n')

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('metrics_md_files', type=str, help="Comma separated list metrics markdown file names")
    argparser.add_argument('--aggregation-json', type=str, required=True, help="Path to a JSON file with metrics to aggregate")
    argparser.add_argument('--gh-pages-link', type=str, required=True, help="Link to this PR's gh-pages directory")
    args = argparser.parse_args()

    aggregations = read_aggregations(args.aggregation_json)
    aggregations = sorted([agg.name for agg in aggregations])
    headers = ["group"] + aggregations

    md_files = args.metrics_md_files.split(',')
    outputs = get_summary_rows(md_files, headers, args.gh_pages_link)

    headers += [agg + "_leaf_agg" for agg in aggregations]
    headers = ["app_log_blowup" if h == "fri.log_blowup" else h for h in headers]
    headers = ["agg_log_blowup" if h == "fri.log_blowup_leaf_agg" else h for h in headers]
    headers += ["instance", "alloc"]

    write_to_md_table(outputs, headers)

if __name__ == '__main__':
    main()