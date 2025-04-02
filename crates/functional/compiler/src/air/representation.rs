use std::collections::HashMap;

use itertools::Itertools;

use crate::{
    air::{
        air::{AirExpression, Bus, Direction, Interaction},
        constructor::AirConstructor,
    },
    core::{
        containers::{DeclarationSet, ExpressionContainer},
        file3::{FlatFunctionCall, FlatMatch, FlatStatement},
        function_resolution::Stage,
        ir::{ArithmeticOperator, BooleanOperator, Expression, Material, StatementVariant, Type},
        scope::ScopePath,
        type_resolution::TypeSet,
    },
    parser::metadata::ParserMetadata,
};

#[derive(Default)]
pub struct Representation {
    pub expressions: Vec<AirExpression>,
    pub owned: Vec<Option<usize>>,
}

impl Representation {
    pub fn add(&mut self, other: &Representation) {
        for (here, there) in self.expressions.iter_mut().zip(other.expressions.iter()) {
            *here = here.plus(there);
        }
    }

    pub fn unowned(&self) -> Representation {
        Representation {
            expressions: self.expressions.clone(),
            owned: vec![None; self.expressions.len()],
        }
    }

    pub fn all_some(options: &[Option<AirExpression>]) -> Self {
        Self {
            expressions: options
                .iter()
                .map(|x| x.as_ref().unwrap().clone())
                .collect(),
            owned: vec![None; options.len()],
        }
    }

    pub fn len(&self) -> usize {
        self.expressions.len()
    }
}

pub struct RepresentationTable<'a> {
    pub(crate) representations: HashMap<(ScopePath, String), Representation>,
    declaration_set: &'a DeclarationSet,
    pub reference_multiplicity_cells: HashMap<usize, usize>,
    pub reference_offsets: HashMap<usize, usize>,
}

impl<'a> RepresentationTable<'a> {
    pub fn new(declaration_set: &'a DeclarationSet) -> Self {
        Self {
            representations: HashMap::new(),
            declaration_set,
            reference_multiplicity_cells: HashMap::new(),
            reference_offsets: HashMap::new(),
        }
    }
    pub fn get_representation(
        &self,
        scope: &ScopePath,
        name: &str,
        type_set: &TypeSet,
    ) -> Vec<AirExpression> {
        let tipo = self.declaration_set.get_declaration_type(scope, name);
        if type_set.calc_type_size(tipo) == 0 {
            return vec![];
        }
        for prefix in scope.prefixes().rev() {
            let representation = self
                .representations
                .get(&(prefix.clone(), name.to_string()));
            if let Some(representation) = representation {
                return representation.expressions.clone();
            }
        }
        unreachable!()
    }
    pub fn fill_in_and_add_representation(
        &mut self,
        air_constructor: &mut AirConstructor,
        scope: &ScopePath,
        name: &String,
        right: &mut [Option<AirExpression>],
    ) {
        let mut representation = Representation::default();
        for right in right.iter_mut() {
            if let Some(right) = right {
                representation.expressions.push(right.clone());
                representation.owned.push(None);
            } else {
                let new_cell = air_constructor.new_normal_cell(scope);
                let expression = AirExpression::single_cell(new_cell);
                representation.expressions.push(expression.clone());
                representation.owned.push(Some(new_cell));
                *right = Some(expression);
            }
        }
        self.insert_representation(scope, name, representation);
    }
    pub fn add_representation(
        &mut self,
        scope: &ScopePath,
        name: &String,
        representation: Vec<AirExpression>,
    ) {
        let representation_len = representation.len();
        self.insert_representation(
            scope,
            name,
            Representation {
                expressions: representation,
                owned: vec![None; representation_len],
            },
        );
    }
    fn insert_representation(
        &mut self,
        scope: &ScopePath,
        name: &str,
        representation: Representation,
    ) {
        let mut ancestor = self.declaration_set.get_declaration_scope(scope, name);
        while ancestor.0.len() < scope.0.len() {
            if let Some(current_representation) = self
                .representations
                .get_mut(&(ancestor.clone(), name.to_string()))
            {
                current_representation.add(&representation);
            } else {
                self.representations.insert(
                    (ancestor.clone(), name.to_string()),
                    representation.unowned(),
                );
            }
            let i = ancestor.0.len();
            ancestor = ancestor.then(scope.0[i].0, scope.0[i].1.clone());
        }
        self.add_scope_specific_representation(scope, name, representation);
    }

    fn add_scope_specific_representation(
        &mut self,
        scope: &ScopePath,
        name: &str,
        representation: Representation,
    ) {
        let prev = self
            .representations
            .insert((scope.clone(), name.to_string()), representation);
        assert!(prev.is_none());
    }

    pub fn new_multiplicity_cell(
        &mut self,
        air_constructor: &mut AirConstructor,
        statement_index: usize,
        scope: &ScopePath,
    ) -> usize {
        let cell = air_constructor.new_normal_cell(scope);
        self.reference_multiplicity_cells
            .insert(statement_index, cell);
        cell
    }

    pub fn set_reference_offset(&mut self, index: usize, offset: usize) {
        self.reference_offsets.insert(index, offset);
    }
}

impl ExpressionContainer {
    pub fn calc_representation(
        &self,
        type_set: &TypeSet,
        representation_table: &RepresentationTable,
        scope: &ScopePath,
    ) -> Vec<AirExpression> {
        match self.expression.as_ref() {
            Expression::Constant { value } => {
                vec![AirExpression::constant(*value)]
            }
            Expression::Variable {
                name, represents, ..
            } => {
                assert!(!*represents);
                representation_table.get_representation(scope, name, type_set)
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let type_name = &type_set.constructors[constructor].0;
                let tipo = &type_set.algebraic_types[type_name];
                let mut result = Vec::new();
                if tipo.variants.len() != 1 {
                    let i = tipo
                        .variants
                        .iter()
                        .position(|variant| &variant.name == constructor)
                        .unwrap();
                    result.push(AirExpression::constant(i as isize));
                }
                for field in fields {
                    result.extend(field.calc_representation(type_set, representation_table, scope));
                }
                let type_length = type_set.calc_type_size(&Type::NamedType(type_name.clone()));
                while result.len() < type_length {
                    result.push(AirExpression::zero());
                }
                result
            }
            Expression::Arithmetic {
                operator,
                left,
                right,
            } => {
                let left = &left.calc_representation(type_set, representation_table, scope)[0];
                let right = &right.calc_representation(type_set, representation_table, scope)[0];
                vec![match operator {
                    ArithmeticOperator::Plus => left.plus(right),
                    ArithmeticOperator::Minus => left.minus(right),
                    ArithmeticOperator::Times => left.times(right),
                    ArithmeticOperator::Div => unreachable!(),
                }]
            }
            Expression::Dematerialized { .. } => vec![],
            Expression::ReadableViewOfPrefix { appendable_prefix } => {
                appendable_prefix.calc_representation(type_set, representation_table, scope)
            }
            Expression::PrefixIntoArray { appendable_prefix } => {
                appendable_prefix.calc_representation(type_set, representation_table, scope)
            }
            Expression::Eq { .. } => unreachable!(),
            Expression::EmptyConstArray { .. } => vec![],
            Expression::ConstArray { elements } => elements
                .iter()
                .flat_map(|element| {
                    element.calc_representation(type_set, representation_table, scope)
                })
                .collect(),
            Expression::ConstArrayConcatenation { left, right } => left
                .calc_representation(type_set, representation_table, scope)
                .into_iter()
                .chain(right.calc_representation(type_set, representation_table, scope))
                .collect(),
            Expression::ConstArrayAccess { array, index } => {
                let array_representation =
                    array.calc_representation(type_set, representation_table, scope);
                let elem_size = type_set.calc_type_size(self.get_type());
                array_representation[index * elem_size..(index + 1) * elem_size].to_vec()
            }
            Expression::ConstArraySlice { array, from, to } => {
                let array_representation =
                    array.calc_representation(type_set, representation_table, scope);
                let (elem_type, _) = array
                    .get_type()
                    .get_const_array_type(Material::Materialized, &ParserMetadata::default())
                    .unwrap();
                let elem_size = type_set.calc_type_size(elem_type);
                array_representation[from * elem_size..to * elem_size].to_vec()
            }
            Expression::ConstArrayRepeated { element, length } => {
                let element_representation =
                    element.calc_representation(type_set, representation_table, scope);
                let elem_type = element.get_type();
                let elem_size = type_set.calc_type_size(elem_type);
                let mut result = Vec::with_capacity(*length * elem_size);
                for _ in 0..*length {
                    result.extend(element_representation.clone());
                }
                result
            }
            Expression::BooleanNot { value } => {
                let value_representation =
                    value.calc_representation(type_set, representation_table, scope);
                vec![AirExpression::one().minus(&value_representation[0])]
            }
            Expression::BooleanBinary {
                left,
                right,
                operator,
            } => {
                let left_representation =
                    left.calc_representation(type_set, representation_table, scope);
                let right_representation =
                    right.calc_representation(type_set, representation_table, scope);
                let x = &left_representation[0];
                let y = &right_representation[0];
                vec![match *operator {
                    BooleanOperator::And => x.times(y),
                    BooleanOperator::Or => x.plus(y).minus(&x.times(y)),
                    BooleanOperator::Xor => x
                        .plus(y)
                        .minus(&AirExpression::constant(2).times(x).times(y)),
                }]
            }
            Expression::Ternary {
                condition,
                true_value,
                false_value,
            } => {
                let condition_representation =
                    condition.calc_representation(type_set, representation_table, scope);
                let condition_expression = &condition_representation[0];
                let true_representation =
                    true_value.calc_representation(type_set, representation_table, scope);
                let false_representation =
                    false_value.calc_representation(type_set, representation_table, scope);
                true_representation
                    .into_iter()
                    .zip_eq(false_representation.into_iter())
                    .map(|(true_expression, false_expression)| {
                        condition_expression.times(&true_expression).plus(
                            &AirExpression::one()
                                .minus(condition_expression)
                                .times(&false_expression),
                        )
                    })
                    .collect()
            }
        }
    }

    pub fn represent_top_down(
        &self,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
        scope: &ScopePath,
        representation: &mut [Option<AirExpression>],
        supplementary: bool, // if supplementary is true then representation should not have None
    ) {
        match self.expression.as_ref() {
            Expression::Constant { value } => {
                if !supplementary {
                    air_constructor.add_scoped_constraint(
                        scope,
                        vec![AirExpression::constant(*value)],
                        representation,
                    );
                }
            }
            Expression::Variable {
                name,
                represents: true,
                ..
            } => {
                if supplementary {
                    representation_table.add_scope_specific_representation(
                        scope,
                        name,
                        Representation::all_some(representation),
                    )
                } else {
                    representation_table.fill_in_and_add_representation(
                        air_constructor,
                        scope,
                        name,
                        representation,
                    );
                }
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let type_name = &type_set.constructors[constructor].0;
                let tipo = &type_set.algebraic_types[type_name];
                let mut offset = 0;
                if tipo.variants.len() != 1 {
                    let i = tipo
                        .variants
                        .iter()
                        .position(|variant| &variant.name == constructor)
                        .unwrap();
                    if !supplementary {
                        air_constructor.add_scoped_constraint(
                            scope,
                            vec![AirExpression::constant(i as isize)],
                            &mut representation[0..1],
                        );
                    }
                    offset += 1;
                }
                for field in fields {
                    let type_length = type_set.calc_type_size(field.get_type());
                    field.represent_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        scope,
                        &mut representation[offset..offset + type_length],
                        supplementary,
                    );
                    offset += type_length;
                }
            }
            Expression::Dematerialized { .. } => {}
            Expression::ConstArray { elements } => {
                let (elem_type, _) = self
                    .get_type()
                    .get_const_array_type(Material::Materialized, &ParserMetadata::default())
                    .unwrap();
                let elem_size = type_set.calc_type_size(elem_type);
                for (i, element) in elements.iter().enumerate() {
                    element.represent_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        scope,
                        &mut representation[i * elem_size..(i + 1) * elem_size],
                        supplementary,
                    );
                }
            }
            _ => {
                if !supplementary {
                    let defined_representation =
                        self.calc_representation(type_set, representation_table, scope);
                    air_constructor.add_scoped_constraint(
                        scope,
                        defined_representation,
                        representation,
                    );
                }
            }
        }
    }

    pub fn create_representation_top_down(
        &self,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
        scope: &ScopePath,
    ) -> Vec<AirExpression> {
        let mut representation = vec![None; type_set.calc_type_size(self.get_type())];
        self.represent_top_down(
            type_set,
            representation_table,
            air_constructor,
            scope,
            &mut representation,
            false,
        );
        representation.into_iter().map(Option::unwrap).collect()
    }

    pub fn represent_top_down_fixed(
        &self,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
        scope: &ScopePath,
        representation: &Vec<AirExpression>,
    ) {
        let mut representation: Vec<_> = representation.iter().cloned().map(Some).collect();
        self.represent_top_down(
            type_set,
            representation_table,
            air_constructor,
            scope,
            &mut representation,
            false,
        );
    }
}

impl FlatMatch {
    pub fn enforce(
        &self,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
    ) {
        if self.check_material == Material::Materialized {
            let type_name = self
                .value
                .get_type()
                .get_named_type(Material::Materialized, &ParserMetadata::default())
                .unwrap();
            let tipo = &type_set.algebraic_types[type_name];

            // make representation of matched value uniform other than variant if possible
            if self.branches.iter().all(|branch| {
                branch
                    .components
                    .iter()
                    .all(|component| component.represents)
            }) {
                let mut representation = vec![None; type_set.calc_type_size(self.value.get_type())];
                if tipo.variants.len() != 1 {
                    let mut variant_number = AirExpression::zero();
                    for (i, branch) in self.branches.iter().enumerate() {
                        let scope = self.scope.then(self.index, branch.constructor.clone());
                        variant_number = variant_number.plus(
                            &air_constructor
                                .get_scope_expression(&scope)
                                .times(&AirExpression::constant(i as isize)),
                        );
                    }
                    representation[0] = Some(variant_number);
                }
                self.value.represent_top_down(
                    type_set,
                    representation_table,
                    air_constructor,
                    &self.scope,
                    &mut representation,
                    false,
                );
                let mut representation =
                    representation.into_iter().map(Option::unwrap).collect_vec();
                for (i, branch) in self.branches.iter().enumerate() {
                    let scope = self.scope.then(self.index, branch.constructor.clone());
                    if tipo.variants.len() == 1 {
                        representation[0] = AirExpression::constant(i as isize);
                        self.value.represent_top_down(
                            type_set,
                            representation_table,
                            air_constructor,
                            &scope,
                            &mut representation.iter().cloned().map(Some).collect_vec(),
                            true,
                        );
                    }
                    let mut offset = if tipo.variants.len() == 1 { 0 } else { 1 };
                    let type_components = &tipo
                        .variants
                        .iter()
                        .find(|variant| variant.name == branch.constructor)
                        .unwrap()
                        .components;
                    for (component, tipo) in branch.components.iter().zip_eq(type_components.iter())
                    {
                        let type_size = type_set.calc_type_size(tipo);
                        if component.represents {
                            representation_table.add_representation(
                                &scope,
                                &component.name,
                                representation[offset..offset + type_size].to_vec(),
                            );
                        }
                        offset += type_size;
                    }
                    air_constructor.add_scoped_constraint(
                        &scope,
                        representation[offset..].to_vec(),
                        &mut vec![Some(AirExpression::zero()); representation.len() - offset],
                    );
                }
            } else {
                for branch in self.branches.iter() {
                    let scope = self.scope.then(self.index, branch.constructor.clone());
                    let mut representation =
                        vec![None; type_set.calc_type_size(self.value.get_type())];
                    if tipo.variants.len() != 1 {
                        let variant_index = tipo
                            .variants
                            .iter()
                            .position(|variant| variant.name == branch.constructor)
                            .unwrap();
                        representation[0] = Some(AirExpression::constant(variant_index as isize));
                    }
                    let type_components = &tipo
                        .variants
                        .iter()
                        .find(|variant| variant.name == branch.constructor)
                        .unwrap()
                        .components;

                    // get from ones that are represented elsewhere
                    let mut offset = if tipo.variants.len() == 1 { 0 } else { 1 };
                    for (component, tipo) in branch.components.iter().zip_eq(type_components.iter())
                    {
                        if !component.represents {
                            let component_representation = representation_table.get_representation(
                                &scope,
                                &component.name,
                                type_set,
                            );
                            for (i, expression) in component_representation.iter().enumerate() {
                                representation[offset + i] = Some(expression.clone());
                            }
                        }
                        let type_size = type_set.calc_type_size(tipo);
                        offset += type_size;
                    }

                    self.value.represent_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &scope,
                        &mut representation,
                        false,
                    );
                    let representation: Vec<_> =
                        representation.into_iter().map(|x| x.unwrap()).collect();

                    let mut offset = if tipo.variants.len() == 1 { 0 } else { 1 };
                    for (component, tipo) in branch.components.iter().zip_eq(type_components.iter())
                    {
                        let type_size = type_set.calc_type_size(tipo);
                        if component.represents {
                            representation_table.add_representation(
                                &scope,
                                &component.name,
                                representation[offset..offset + type_size].to_vec(),
                            );
                        }
                        offset += type_size;
                    }
                    air_constructor.add_scoped_constraint(
                        &scope,
                        representation[offset..].to_vec(),
                        &mut vec![Some(AirExpression::zero()); representation.len() - offset],
                    );
                }
            }
        }
    }
}

impl FlatFunctionCall {
    pub fn represent_stage(
        &self,
        interaction: &mut Interaction,
        stage: Stage,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
    ) {
        for i in stage.start..stage.mid {
            let argument = &self.arguments[i];
            interaction.fields.extend(argument.calc_representation(
                type_set,
                representation_table,
                &self.scope,
            ));
        }
        for i in stage.mid..stage.end {
            interaction
                .fields
                .extend(self.arguments[i].create_representation_top_down(
                    type_set,
                    representation_table,
                    air_constructor,
                    &self.scope,
                ));
        }
    }
}

impl FlatStatement {
    pub fn enforce(
        &self,
        index: usize,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
    ) {
        if self.material == Material::Materialized {
            match &self.statement {
                StatementVariant::VariableDeclaration {
                    name,
                    tipo,
                    represents,
                } => {
                    if *represents {
                        representation_table.fill_in_and_add_representation(
                            air_constructor,
                            &self.scope,
                            name,
                            &mut vec![None; type_set.calc_type_size(tipo)],
                        );
                    }
                }
                StatementVariant::Equality { left, right } => {
                    let representation =
                        right.calc_representation(type_set, representation_table, &self.scope);
                    let mut representation: Vec<_> = representation.into_iter().map(Some).collect();
                    left.represent_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                        &mut representation,
                        false,
                    );
                }
                StatementVariant::Reference { reference, data } => {
                    let data_representation =
                        data.calc_representation(type_set, representation_table, &self.scope);
                    let reference_representation =
                        vec![air_constructor.get_reference_address_expression(index)];
                    reference.represent_top_down_fixed(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                        &reference_representation,
                    );
                    let multiplicity_cell = representation_table.new_multiplicity_cell(
                        air_constructor,
                        index,
                        &self.scope,
                    );
                    let mut fields = vec![];
                    fields.extend(reference_representation);
                    fields.extend(data_representation);
                    air_constructor.add_scoped_interaction(
                        &self.scope,
                        Interaction {
                            bus: Bus::Reference,
                            direction: Direction::Receive,
                            multiplicity: AirExpression::single_cell(multiplicity_cell),
                            fields,
                        },
                    );
                }
                StatementVariant::Dereference { data, reference } => {
                    let reference_representation =
                        reference.calc_representation(type_set, representation_table, &self.scope);
                    let data_representation = data.create_representation_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                    );
                    let mut fields = vec![];
                    fields.extend(reference_representation);
                    fields.extend(data_representation);
                    air_constructor.add_scoped_interaction(
                        &self.scope,
                        Interaction {
                            bus: Bus::Reference,
                            direction: Direction::Send,
                            multiplicity: AirExpression::one(),
                            fields,
                        },
                    );
                }
                StatementVariant::EmptyPrefix { prefix: array, .. } => {
                    let representation =
                        vec![air_constructor.get_reference_address_expression(index)];
                    array.represent_top_down_fixed(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                        &representation,
                    );
                }
                StatementVariant::PrefixAppend {
                    new_prefix,
                    elem: _,
                    old_prefix,
                } => {
                    let array_representation =
                        old_prefix.calc_representation(type_set, representation_table, &self.scope);

                    new_prefix.represent_top_down_fixed(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                        &array_representation,
                    );
                }
                StatementVariant::ArrayAccess { array, index, elem } => {
                    let array_representation =
                        array.calc_representation(type_set, representation_table, &self.scope);
                    let index_representation =
                        index.calc_representation(type_set, representation_table, &self.scope);
                    let elem_representation = elem.create_representation_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                    );
                    let mut fields = vec![];
                    fields.extend(array_representation);
                    fields.extend(index_representation);
                    fields.extend(elem_representation);
                    air_constructor.add_scoped_interaction(
                        &self.scope,
                        Interaction {
                            bus: Bus::Array,
                            direction: Direction::Send,
                            multiplicity: AirExpression::one(),
                            fields,
                        },
                    );
                }
            }
        }
    }
}
