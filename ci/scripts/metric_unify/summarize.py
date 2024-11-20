import argparse
import re

from main import read_aggregations;

COLS_PER_SECTION = 4

class MdTableCell:
    val = 0
    diff = None
    def __init__(self, val, diff):
        self.val = val
        self.diff = diff
    def __str__(self):
        div = f"{self.val:,}"
        if self.diff is not None:
            color = "red" if self.diff > 0 else "green"
            original_val = self.val - self.diff
            diff_percent = float(self.diff) / float(original_val)
            span = f"{self.diff:+,} [{diff_percent:+.1%}]"
            return format_cell(div, span, color)
        return format_cell(div, None, None)
    def __repr__(self):
        return self.__str__()
    def __iadd__(self, other):
        assert(isinstance(other, MdTableCell))
        self.val += other.val
        if self.diff is not None or other.diff is not None:
            self.diff = (self.diff if self.diff is not None else 0) + (other.diff if other.diff is not None else 0)
        return self

class MdTableRow:
    group = ""
    log_blowup = None;
    cells = []
    def __init__(self, group, log_blowup, cells):
        self.group = group
        self.log_blowup = log_blowup
        self.cells = cells
        assert(len(self.cells) == COLS_PER_SECTION - 1)
    def __str__(self):
        return f"{self.group}-{self.log_blowup.val}-{self.cells}"
    def __repr__(self):
        return self.__str__()
    def __iadd__(self, other):
        for i in range(len(self.cells)):
            self.cells[i] += other.cells[i]
        return self

def format_cell(div, span = None, span_color = None):
    ret = ""
    if span is not None:
        ret += f"<span style='color: {span_color}'>({span})</span>"
    ret += f"<div style='text-align: right'> {div} </div> "
    return ret

def md_to_cell(text):
    val_match = re.search(r"<div.*?>([\d,.]+)</div>", text)
    if val_match:
        val_str = val_match.group(1)
        decimal = '.' in val_str
        val = float(val_str.replace(',', '')) if decimal else int(val_str.replace(',', ''))
    else:
        val = None
        decimal = False
    diff_match = re.search(r'<span.*?>\(([-+]?\d[\d,]*)', text)
    diff = (float(diff_match.group(1).replace(',', '')) if decimal else int(diff_match.group(1).replace(',', ''))) if diff_match else None
    return MdTableCell(val, diff)

def read_first_markdown_table(md_file):
    with open(md_file, 'r') as file:
        lines = iter(file.readlines())
    next(lines)
    next(lines)

    table = []
    for line in lines:
        if not line.startswith('|'):
            break
        else:
            cols = [col.strip() for col in line.split('|') if col.strip()]
            name = cols[0]
            log_blowup = md_to_cell(cols[1])
            cells = [md_to_cell(col) for col in cols[2:]]
            table.append(MdTableRow(name, log_blowup, cells))

    other_info = []
    for line in lines:
        if "Instance Type:" in line:
            other_info += [f"{line.split(':')[1].strip()}"]
            continue

        if "Memory Allocator:" in line:
            other_info += [f"{line.split(':')[1].strip()}"]
            break
    return table, other_info

# group stats (ex. app, agg, root) are either added to the row as is, or aggregated together
def generate_row(md_file, sections, aggregation_groups, gh_pages_link):
    table, other_info = read_first_markdown_table(md_file)
    sections = [table[0].group] + sections
    section_by_group = {};
    for row in table:
        if row.group in sections:
            section_by_group[row.group] = row
            continue
        for group_pattern, group_name in aggregation_groups.items():
            if re.search(group_pattern, row.group):
                if group_name in section_by_group:
                    section_by_group[group_name] += row
                else:
                    section_by_group[group_name] = row
                break

    res = [f"[ {table[0].group} ]({gh_pages_link}/individual/{md_file})"]
    for section in sections:
        group_row = section_by_group.get(section)
        if group_row is None:
            res += [format_cell("-", None, None)] * COLS_PER_SECTION
        else:    
            res.append(str(group_row.log_blowup))
            for cell in group_row.cells:
                res.append(str(cell))
    res += other_info
    return res

def write_md_table(rows, headers, file_path='summary.md', rewrite=False):
    with open(file_path, 'w' if rewrite else 'a') as f:
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '---|' * len(headers) + '\n')
        for row in rows:
            f.write('| ' + ' | '.join(row) + ' |\n')
        f.write('\n')

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('metrics_md_files', type=str, help="Comma separated list of metrics markdown file names")
    argparser.add_argument('--e2e-md-files', type=str, required=False, help="Comma separated list of e2e metrics markdown file names")
    argparser.add_argument('--aggregation-json', type=str, required=True, help="Path to a JSON file with metrics to aggregate")
    argparser.add_argument('--gh-pages-link', type=str, required=True, help="Link to this PR's gh-pages directory")
    args = argparser.parse_args()

    aggregations = read_aggregations(args.aggregation_json)
    aggregations = sorted([agg.name.replace("fri.", "") for agg in aggregations])
    headers = ["group"] + ["app_" + agg for agg in aggregations] + ["leaf_" + agg for agg in aggregations]
    e2e_headers = headers + ["root_" + agg for agg in aggregations] + ["internal_" + agg for agg in aggregations] + ["instance", "alloc"]
    headers += ["instance", "alloc"]

    md_files = args.metrics_md_files.split(',')
    outputs = []
    for md_file in md_files:
        outputs.append(generate_row(md_file, ["leaf_aggregation"], {}, args.gh_pages_link))
    write_md_table(outputs, headers, rewrite=True)

    if args.e2e_md_files is not None:
        outputs = []
        md_files = args.e2e_md_files.split(',')
        for md_file in md_files:
            outputs.append(generate_row(md_file, ["root_verifier", "leaf_verifier", "internal_verifier"], {"internal.*": "internal_verifier"}, args.gh_pages_link))
        write_md_table(outputs, e2e_headers)

if __name__ == '__main__':
    main()