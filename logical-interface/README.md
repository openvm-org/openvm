# AFS Logical Interface

The AFS Logical Interface allows users to specify simple data types (such as u64, u128, etc) that want to use to the underlying database, which potentially holds data in much larger types (bytes32, bytes1024, etc).

## Data storage

Data is stored in tables, referenced by `TableId`, in a MockDb. The current implementation is essentially just a mapping of `TableId`s to HashMaps of (index, data), with some metadata that holds the size of the data.

## Component diagram

```bash
    [ --AfsInterface--- ]
        |          ^
        |          |
        |     [ -Table- ]
        |          ^
        v          |
    [ -FixedBytesCodec- ]
             ^
             |
             v
    [ -----MockDb------ ]
```

## Usage

Users can instantiate `AfsInterface` and either manually create a table or load instructions from a file.

### Creating a database table

Users can create tables and insert/write to those tables. It is an error to try to create a new table with an ID that already exists.

### Insert/write

Users can insert (index does not exist in the table) or write (index exists, data is overwritten) to the database.

### Reading

Data reads are done by getting a `Table` and reading from it. This is described below.

## Table

The `Table` object is read-only and converts the underlying database types to simple data types (u64, u128, etc) and is valid only for the moment in time that it is requested. Once additional writes happen to the underlying database, the current `Table` object will no longer reflect the underlying database and a user must call `get_table` again to get the latest data.
